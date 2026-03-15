import Photos
import UIKit

struct SelectedPhoto {
    let image: UIImage
    let originalPixelSize: CGSize
}

extension Notification.Name {
    static let photoLibraryServiceDidUpdate = Notification.Name("PhotoLibraryServiceDidUpdate")
    static let photoAnalysisCoordinatorDidUpdate = Notification.Name("PhotoAnalysisCoordinatorDidUpdate")
}

@MainActor
final class PhotoLibraryService: NSObject, PHPhotoLibraryChangeObserver {
    static let shared = PhotoLibraryService()

    private let imageManager = PHCachingImageManager()
    private var assets: [PHAsset] = []

    override init() {
        super.init()
        PHPhotoLibrary.shared().register(self)
        refreshAssetsIfAuthorized()
    }

    deinit {
        PHPhotoLibrary.shared().unregisterChangeObserver(self)
    }

    var authorizationStatus: PHAuthorizationStatus {
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    }

    var hasFullAccess: Bool {
        authorizationStatus == .authorized
    }

    func currentAssets() -> [PHAsset] {
        assets
    }

    func asset(withLocalIdentifier identifier: String) -> PHAsset? {
        assets.first { $0.localIdentifier == identifier }
    }

    func requestFullAccess() async -> Bool {
        switch authorizationStatus {
        case .authorized:
            refreshAssetsIfAuthorized()
            return true
        case .notDetermined:
            let status = await withCheckedContinuation { continuation in
                PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                    continuation.resume(returning: newStatus)
                }
            }
            refreshAssetsIfAuthorized()
            return status == .authorized
        default:
            refreshAssetsIfAuthorized()
            return false
        }
    }

    func refreshAssetsIfAuthorized() {
        guard hasFullAccess else {
            assets = []
            notifyDidUpdate()
            return
        }

        let options = PHFetchOptions()
        options.sortDescriptors = [
            NSSortDescriptor(key: "creationDate", ascending: false),
            NSSortDescriptor(key: "modificationDate", ascending: false),
        ]
        let result = PHAsset.fetchAssets(with: .image, options: options)

        var snapshot: [PHAsset] = []
        snapshot.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in
            snapshot.append(asset)
        }

        assets = snapshot
        print("[PhotoLibraryService] refreshed assets, count=\(assets.count)")
        notifyDidUpdate()
    }

    func requestThumbnail(
        for asset: PHAsset,
        targetSize: CGSize,
        contentMode: PHImageContentMode = .aspectFill,
        completion: @escaping @MainActor (UIImage?) -> Void
    ) -> PHImageRequestID {
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true

        return imageManager.requestImage(
            for: asset,
            targetSize: targetSize,
            contentMode: contentMode,
            options: options
        ) { image, _ in
            Task { @MainActor in
                completion(image)
            }
        }
    }

    func cancelImageRequest(_ requestID: PHImageRequestID) {
        imageManager.cancelImageRequest(requestID)
    }

    func startCaching(for assets: [PHAsset], targetSize: CGSize, contentMode: PHImageContentMode = .aspectFill) {
        imageManager.startCachingImages(for: assets, targetSize: targetSize, contentMode: contentMode, options: nil)
    }

    func stopCaching(for assets: [PHAsset], targetSize: CGSize, contentMode: PHImageContentMode = .aspectFill) {
        imageManager.stopCachingImages(for: assets, targetSize: targetSize, contentMode: contentMode, options: nil)
    }

    func requestAnalysisPhoto(for asset: PHAsset, userInterfaceIdiom: UIUserInterfaceIdiom) async throws -> SelectedPhoto {
        let requestOptions = PHImageRequestOptions()
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.resizeMode = .fast
        requestOptions.isNetworkAccessAllowed = true
        requestOptions.isSynchronous = false

        let longEdge = LocalVLMService.preferredPhotoRequestLongEdge(for: userInterfaceIdiom)
        let aspectRatio = CGFloat(asset.pixelWidth) / max(CGFloat(asset.pixelHeight), 1)
        let targetSize: CGSize
        if aspectRatio >= 1 {
            targetSize = CGSize(width: longEdge, height: round(longEdge / aspectRatio))
        } else {
            targetSize = CGSize(width: round(longEdge * aspectRatio), height: longEdge)
        }

        print(
            "[PhotoLibraryService] requestAnalysisPhoto asset=\(asset.localIdentifier) "
                + "original=\(asset.pixelWidth)x\(asset.pixelHeight) target=\(Int(targetSize.width))x\(Int(targetSize.height))"
        )

        return try await withCheckedThrowingContinuation { continuation in
            imageManager.requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFit,
                options: requestOptions
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true {
                    return
                }
                guard let image else {
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }

                continuation.resume(
                    returning: SelectedPhoto(
                        image: image,
                        originalPixelSize: CGSize(width: asset.pixelWidth, height: asset.pixelHeight)
                    )
                )
            }
        }
    }

    func requestOCRImageData(for asset: PHAsset) async throws -> Data {
        let requestOptions = PHImageRequestOptions()
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.resizeMode = .none
        requestOptions.isNetworkAccessAllowed = true
        requestOptions.isSynchronous = false
        requestOptions.version = .current

        print("[PhotoLibraryService] requestOCRImageData asset=\(asset.localIdentifier) original=\(asset.pixelWidth)x\(asset.pixelHeight)")

        return try await withCheckedThrowingContinuation { continuation in
            imageManager.requestImageDataAndOrientation(for: asset, options: requestOptions) { data, _, _, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                guard let data, !data.isEmpty else {
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                print("[PhotoLibraryService] requestOCRImageData bytes=\(data.count)")
                continuation.resume(returning: data)
            }
        }
    }

    nonisolated func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task { @MainActor in
            self.refreshAssetsIfAuthorized()
        }
    }

    private func notifyDidUpdate() {
        NotificationCenter.default.post(name: .photoLibraryServiceDidUpdate, object: nil)
    }
}
