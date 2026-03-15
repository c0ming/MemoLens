import Foundation
import Photos
import UIKit
import WidgetKit

@MainActor
final class FeaturedMemoryWidgetSync {
    static let shared = FeaturedMemoryWidgetSync()

    private static let widgetImageTargetSize = CGSize(width: 1200, height: 1200)
    private static let widgetImageMaxPixelArea: CGFloat = 1_800_000
    private static let widgetImageMaxLongEdge: CGFloat = 1_600
    private static let simulatorCaption = "湖面安静下来时，午后的光也慢慢落进了回忆里。"

    private let imageManager = PHCachingImageManager()

    private init() {}

    func refresh(records: [PhotoAnalysisRecord], assets: [PHAsset]) async {
        #if targetEnvironment(simulator)
        await refreshSimulatorPreview()
        return
        #endif

        do {
            guard let selection = selectFeaturedMemory(from: records, assets: assets) else {
                try writeEmptyPayload()
                reloadWidgetTimelines()
                return
            }

            let image = try await requestWidgetImage(for: selection.asset)
            let imageData = preparedWidgetImage(from: image).normalizedJPEGData()
            let imageURL = try MemoryWidgetShared.imageURL()
            try imageData.write(to: imageURL, options: [.atomic])

            let payload = MemoryWidgetPayload(
                state: .ready,
                assetLocalIdentifier: selection.record.assetLocalIdentifier,
                captionLine: selection.captionLine,
                imageFileName: MemoryWidgetShared.imageFileName,
                photoTimestamp: selection.asset.creationDate?.timeIntervalSince1970
                    ?? selection.asset.modificationDate?.timeIntervalSince1970,
                updatedAt: Date().timeIntervalSince1970
            )
            try writePayload(payload)
            print("[FeaturedMemoryWidgetSync] wrote featured widget payload for \(selection.record.assetLocalIdentifier)")
        } catch {
            print("[FeaturedMemoryWidgetSync] refresh failed: \(error.localizedDescription)")
        }

        reloadWidgetTimelines()
    }

    #if targetEnvironment(simulator)
    func refreshSimulatorPreview() async {
        do {
            guard
                let imageURL = Bundle.main.url(forResource: "VLTestInput", withExtension: "jpg"),
                let image = UIImage(contentsOfFile: imageURL.path)
            else {
                try writeEmptyPayload()
                reloadWidgetTimelines()
                return
            }

            let widgetImageURL = try MemoryWidgetShared.imageURL()
            try preparedWidgetImage(from: image).normalizedJPEGData().write(to: widgetImageURL, options: [.atomic])

            let payload = MemoryWidgetPayload(
                state: .ready,
                assetLocalIdentifier: "simulator-preview",
                captionLine: Self.simulatorCaption,
                imageFileName: MemoryWidgetShared.imageFileName,
                photoTimestamp: Date().timeIntervalSince1970,
                updatedAt: Date().timeIntervalSince1970
            )
            try writePayload(payload)
            print("[FeaturedMemoryWidgetSync] wrote simulator preview payload")
        } catch {
            print("[FeaturedMemoryWidgetSync] simulator preview failed: \(error.localizedDescription)")
        }

        reloadWidgetTimelines()
    }
    #endif

    private func reloadWidgetTimelines() {
        WidgetCenter.shared.reloadTimelines(ofKind: MemoryWidgetShared.widgetKind)
        WidgetCenter.shared.reloadAllTimelines()
    }

    private func selectFeaturedMemory(
        from records: [PhotoAnalysisRecord],
        assets: [PHAsset]
    ) -> (record: PhotoAnalysisRecord, asset: PHAsset, captionLine: String)? {
        let assetByIdentifier = Dictionary(uniqueKeysWithValues: assets.map { ($0.localIdentifier, $0) })

        let candidates = records.compactMap { record -> (PhotoAnalysisRecord, PHAsset, String)? in
            guard record.status == .completed else { return nil }
            guard let asset = assetByIdentifier[record.assetLocalIdentifier] else { return nil }
            guard let captionLine = extractCaptionLine(from: record.resultJSONString), !captionLine.isEmpty else { return nil }
            return (record, asset, captionLine)
        }

        return candidates.sorted { lhs, rhs in
            let lhsScore = lhs.0.memoryScore ?? -1
            let rhsScore = rhs.0.memoryScore ?? -1
            if abs(lhsScore - rhsScore) > 0.01 {
                return lhsScore > rhsScore
            }
            return lhs.0.updatedAt > rhs.0.updatedAt
        }.first
    }

    private func requestWidgetImage(for asset: PHAsset) async throws -> UIImage {
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        options.isSynchronous = false

        return try await withCheckedThrowingContinuation { continuation in
            imageManager.requestImage(
                for: asset,
                targetSize: Self.widgetImageTargetSize,
                contentMode: .aspectFill,
                options: options
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    continuation.resume(throwing: error)
                    return
                }
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    continuation.resume(throwing: CocoaError(.userCancelled))
                    return
                }
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true {
                    return
                }
                guard let image else {
                    continuation.resume(throwing: CocoaError(.coderReadCorrupt))
                    return
                }
                continuation.resume(returning: image)
            }
        }
    }

    private func preparedWidgetImage(from image: UIImage) -> UIImage {
        image.resizedToFit(
            maxPixelArea: Self.widgetImageMaxPixelArea,
            maxLongEdge: Self.widgetImageMaxLongEdge
        )
    }

    private func extractCaptionLine(from resultJSONString: String?) -> String? {
        guard
            let resultJSONString,
            let data = resultJSONString.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        let caption = (object["cl"] as? String) ?? (object["caption_line"] as? String)
        return caption?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func writePayload(_ payload: MemoryWidgetPayload) throws {
        let payloadURL = try MemoryWidgetShared.payloadURL()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(payload)
        try data.write(to: payloadURL, options: [.atomic])
    }

    private func writeEmptyPayload() throws {
        let payload = MemoryWidgetPayload(
            state: .empty,
            assetLocalIdentifier: nil,
            captionLine: "还没有可展示的照片",
            imageFileName: nil,
            photoTimestamp: nil,
            updatedAt: Date().timeIntervalSince1970
        )
        try writePayload(payload)

        let imageURL = try MemoryWidgetShared.imageURL()
        if FileManager.default.fileExists(atPath: imageURL.path) {
            try? FileManager.default.removeItem(at: imageURL)
        }
    }
}
