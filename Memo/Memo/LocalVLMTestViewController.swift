//
//  LocalVLMTestViewController.swift
//  Memo
//
//  Created by c0mingxx on 2026/3/14.
//

import Photos
import PhotosUI
import SnapKit
import UIKit

class LocalVLMTestViewController: UIViewController, PHPickerViewControllerDelegate {
    private let scrollView = UIScrollView()
    private let contentView = UIView()
    private let titleLabel = UILabel()
    private let subtitleLabel = UILabel()
    private let imageView = UIImageView()
    private let runButton = UIButton(type: .system)
    private let statusLabel = UILabel()
    private let outputView = UITextView()
    private let placeholderLabel = UILabel()
    private let spinner = UIActivityIndicatorView(style: .large)
    private var hasReceivedChunk = false
    private var runStartedAt = Date()
    private var latestMetrics: VLMRunMetrics?
    private var selectedOriginalPixelSize: CGSize?

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "VL 测试"
        Self.log("viewDidLoad")
        configureUI()
    }

    private func configureUI() {
        view.backgroundColor = .systemBackground

        scrollView.alwaysBounceVertical = true
        scrollView.contentInsetAdjustmentBehavior = .always

        titleLabel.text = "Qwen3-VL 2B 本地测试"
        titleLabel.font = .systemFont(ofSize: 28, weight: .bold)
        titleLabel.numberOfLines = 0

        subtitleLabel.text = "点击下方预览区域选择系统相册中的照片，再用本地 2B VL 模型和 Python 同款 JSON 提示词做视觉分析。"
        subtitleLabel.font = .systemFont(ofSize: 15, weight: .regular)
        subtitleLabel.textColor = .secondaryLabel
        subtitleLabel.numberOfLines = 0

        imageView.contentMode = .scaleAspectFill
        imageView.layer.cornerRadius = 20
        imageView.layer.masksToBounds = true
        imageView.backgroundColor = .secondarySystemBackground
        imageView.isUserInteractionEnabled = true

        placeholderLabel.text = "点按这里选择系统相册中的照片"
        placeholderLabel.font = .systemFont(ofSize: 16, weight: .medium)
        placeholderLabel.textColor = .secondaryLabel
        placeholderLabel.textAlignment = .center
        placeholderLabel.numberOfLines = 0

        runButton.configuration = .filled()
        runButton.configuration?.title = "分析所选照片"
        runButton.isEnabled = false
        runButton.addTarget(self, action: #selector(runVLTest), for: .touchUpInside)

        statusLabel.font = .systemFont(ofSize: 14, weight: .medium)
        statusLabel.textColor = .secondaryLabel
        statusLabel.numberOfLines = 0
        statusLabel.text = "待命。先选照片，模型会在第一次分析时加载。"

        outputView.font = .monospacedSystemFont(ofSize: 14, weight: .regular)
        outputView.isEditable = false
        outputView.isScrollEnabled = true
        outputView.backgroundColor = .secondarySystemBackground
        outputView.layer.cornerRadius = 16
        outputView.textContainerInset = UIEdgeInsets(top: 16, left: 14, bottom: 16, right: 14)
        outputView.text = "这里会显示 VL 输出。"

        spinner.hidesWhenStopped = true

        view.addSubview(scrollView)
        scrollView.addSubview(contentView)

        imageView.addSubview(placeholderLabel)
        imageView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(selectPhoto)))

        [titleLabel, subtitleLabel, imageView, runButton, statusLabel, outputView, spinner].forEach {
            contentView.addSubview($0)
        }

        scrollView.snp.makeConstraints { make in
            make.edges.equalTo(view.safeAreaLayoutGuide)
        }

        contentView.snp.makeConstraints { make in
            make.edges.equalTo(scrollView.contentLayoutGuide)
            make.width.equalTo(scrollView.frameLayoutGuide)
        }

        titleLabel.snp.makeConstraints { make in
            make.top.equalTo(contentView).inset(24)
            make.leading.trailing.equalTo(contentView).inset(20)
        }

        subtitleLabel.snp.makeConstraints { make in
            make.top.equalTo(titleLabel.snp.bottom).offset(10)
            make.leading.trailing.equalTo(titleLabel)
        }

        imageView.snp.makeConstraints { make in
            make.top.equalTo(subtitleLabel.snp.bottom).offset(20)
            make.centerX.equalTo(view)
            make.width.equalTo(titleLabel.snp.width).multipliedBy(0.72)
            make.height.equalTo(imageView.snp.width).multipliedBy(4.0 / 3.0)
        }

        placeholderLabel.snp.makeConstraints { make in
            make.center.equalTo(imageView)
            make.leading.trailing.equalTo(imageView).inset(20)
        }

        runButton.snp.makeConstraints { make in
            make.top.equalTo(imageView.snp.bottom).offset(20)
            make.leading.trailing.equalTo(titleLabel)
            make.height.equalTo(52)
        }

        statusLabel.snp.makeConstraints { make in
            make.top.equalTo(runButton.snp.bottom).offset(12)
            make.leading.trailing.equalTo(titleLabel)
        }

        outputView.snp.makeConstraints { make in
            make.top.equalTo(statusLabel.snp.bottom).offset(12)
            make.leading.trailing.equalTo(titleLabel)
            make.height.equalTo(320)
            make.bottom.equalTo(contentView).inset(20)
        }

        spinner.snp.makeConstraints { make in
            make.center.equalTo(imageView)
        }
    }

    @objc
    private func selectPhoto() {
        Self.log("selectPhoto tapped")
        Task { [weak self] in
            guard let self else { return }
            let granted = await self.ensureFullPhotoLibraryAccess()
            Self.log("photo access granted=\(granted)")
            guard granted else { return }
            await MainActor.run {
                var configuration = PHPickerConfiguration(photoLibrary: .shared())
                configuration.filter = .images
                configuration.selectionLimit = 1

                let picker = PHPickerViewController(configuration: configuration)
                picker.delegate = self
                Self.log("presenting PHPickerViewController")
                self.present(picker, animated: true)
            }
        }
    }

    @objc
    private func runVLTest() {
        guard let selectedImage = imageView.image else {
            statusLabel.text = "请先从系统相册选择一张照片。"
            Self.log("runVLTest aborted, no selected image")
            return
        }
        Self.log("runVLTest start, imageSize=\(selectedImage.size.width)x\(selectedImage.size.height)")
        hasReceivedChunk = false
        latestMetrics = nil
        runStartedAt = Date()
        imageView.isUserInteractionEnabled = false
        runButton.isEnabled = false
        spinner.startAnimating()
        statusLabel.text = "正在加载 2B VL 模型并流式生成..."
        outputView.text = "生成中..."

        let selectedImageData = selectedImage.normalizedJPEGData()
        Self.log("runVLTest image encoded, bytes=\(selectedImageData.count)")

        Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await LocalVLMService.shared.streamImage(
                    selectedImageData,
                    userInterfaceIdiom: self.traitCollection.userInterfaceIdiom,
                    originalPixelSize: self.selectedOriginalPixelSize,
                    onLoadProgress: { [weak self] progress in
                        guard let self else { return }
                        await MainActor.run {
                            if !self.hasReceivedChunk {
                                self.statusLabel.text = "正在加载模型 \(Int(progress * 100))%..."
                            }
                        }
                    },
                    onImageInfo: { [weak self] imageInfo in
                        guard let self else { return }
                        await MainActor.run {
                            self.statusLabel.text =
                                "所选照片原始 \(imageInfo.originalWidth)x\(imageInfo.originalHeight)，输入 \(imageInfo.effectiveWidth)x\(imageInfo.effectiveHeight)"
                        }
                    },
                    onChunk: { [weak self] chunk in
                        guard let self else { return }
                        await MainActor.run {
                            if !self.hasReceivedChunk {
                                self.outputView.text = ""
                                self.hasReceivedChunk = true
                            }
                            self.outputView.text += chunk
                            self.statusLabel.text = "流式生成中... 已耗时 \(self.formatSeconds(Date().timeIntervalSince(self.runStartedAt)))"
                            let range = NSRange(location: max(self.outputView.text.count - 1, 0), length: 1)
                            self.outputView.scrollRangeToVisible(range)
                        }
                    },
                    onComplete: { [weak self] metrics in
                        guard let self else { return }
                        await MainActor.run {
                            self.latestMetrics = metrics
                        }
                    }
                )
                await MainActor.run {
                    Self.log("runVLTest completed, resultChars=\(result.count)")
                    self.outputView.text = result
                    if let metrics = self.latestMetrics {
                        Self.log(
                            "runVLTest metrics, generatedTokens=\(metrics.generatedTokenCount), totalTime=\(metrics.totalTime), tps=\(metrics.tokensPerSecond)"
                        )
                        self.statusLabel.text = "生成完成。输出 \(metrics.generatedTokenCount) tokens，耗时 \(self.formatSeconds(metrics.totalTime))，\(self.formatRate(metrics.tokensPerSecond)) tok/s"
                    } else {
                        Self.log("runVLTest completed without metrics")
                        self.statusLabel.text = "流式生成完成。"
                    }
                    self.imageView.isUserInteractionEnabled = true
                    self.runButton.isEnabled = true
                    self.spinner.stopAnimating()
                }
            } catch {
                await MainActor.run {
                    Self.log("runVLTest failed: \(error.localizedDescription)")
                    self.outputView.text = error.localizedDescription
                    self.statusLabel.text = "生成失败。"
                    self.imageView.isUserInteractionEnabled = true
                    self.runButton.isEnabled = true
                    self.spinner.stopAnimating()
                }
            }
        }
    }

    private func formatSeconds(_ seconds: TimeInterval) -> String {
        String(format: "%.1fs", seconds)
    }

    private func formatRate(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    private func ensureFullPhotoLibraryAccess() async -> Bool {
        let currentStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        Self.log("photo authorization status=\(currentStatus.rawValue)")
        switch currentStatus {
        case .authorized:
            return true
        case .notDetermined:
            let newStatus = await requestPhotoLibraryAuthorization()
            Self.log("photo authorization requested, newStatus=\(newStatus.rawValue)")
            if newStatus == .authorized {
                return true
            }
            await MainActor.run {
                self.presentPhotoAccessAlert(for: newStatus)
            }
            return false
        case .limited, .denied, .restricted:
            await MainActor.run {
                self.presentPhotoAccessAlert(for: currentStatus)
            }
            return false
        @unknown default:
            return false
        }
    }

    private func requestPhotoLibraryAuthorization() async -> PHAuthorizationStatus {
        await withCheckedContinuation { continuation in
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
                continuation.resume(returning: status)
            }
        }
    }

    @MainActor
    private func presentPhotoAccessAlert(for status: PHAuthorizationStatus) {
        Self.log("presentPhotoAccessAlert status=\(status.rawValue)")
        let message: String
        switch status {
        case .limited:
            message = "当前只给了“有限照片访问”。这个页面要求“完全访问”系统相册后再选图。"
        case .denied, .restricted:
            message = "请在系统设置里把照片权限改为“完全访问”，然后再回来选图。"
        default:
            message = "需要系统相册的完全访问权限后才能选图。"
        }

        statusLabel.text = "需要系统相册完全访问权限。"
        runButton.isEnabled = false

        let alert = UIAlertController(title: "需要完全访问相册", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "取消", style: .cancel))
        alert.addAction(
            UIAlertAction(title: "去设置", style: .default) { _ in
                guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
                UIApplication.shared.open(url)
            }
        )
        present(alert, animated: true)
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        Self.log("picker didFinishPicking, count=\(results.count)")
        picker.dismiss(animated: true)

        guard let result = results.first else {
            Self.log("picker returned no selection")
            return
        }

        statusLabel.text = "正在读取所选照片..."
        outputView.text = "这里会显示 VL 输出。"
        imageView.isUserInteractionEnabled = false
        runButton.isEnabled = false
        spinner.startAnimating()

        Task { [weak self] in
            guard let self else { return }
            do {
                let selectedPhoto = try await self.loadSelectedPhoto(from: result)
                await MainActor.run {
                    Self.log(
                        "selected photo ready, previewSize=\(selectedPhoto.image.size.width)x\(selectedPhoto.image.size.height), "
                            + "originalSize=\(selectedPhoto.originalPixelSize.width)x\(selectedPhoto.originalPixelSize.height)"
                    )
                    self.selectedOriginalPixelSize = selectedPhoto.originalPixelSize
                    self.imageView.image = selectedPhoto.image
                    self.placeholderLabel.isHidden = self.imageView.image != nil
                    self.statusLabel.text = "照片已选中。点击下方按钮开始分析。"
                    self.imageView.isUserInteractionEnabled = true
                    self.runButton.isEnabled = true
                    self.spinner.stopAnimating()
                }
            } catch {
                await MainActor.run {
                    Self.log("loadSelectedPhoto failed: \(error.localizedDescription)")
                    self.selectedOriginalPixelSize = nil
                    self.imageView.image = nil
                    self.placeholderLabel.isHidden = false
                    self.statusLabel.text = "读取照片失败。"
                    self.outputView.text = error.localizedDescription
                    self.imageView.isUserInteractionEnabled = true
                    self.runButton.isEnabled = false
                    self.spinner.stopAnimating()
                }
            }
        }
    }

    private func loadSelectedPhoto(from result: PHPickerResult) async throws -> SelectedPhoto {
        if let assetIdentifier = result.assetIdentifier {
            Self.log("loading selected photo from PHAsset, assetIdentifier=\(assetIdentifier)")
            let assets = PHAsset.fetchAssets(withLocalIdentifiers: [assetIdentifier], options: nil)
            if let asset = assets.firstObject {
                return try await loadSelectedPhoto(from: asset)
            }
            Self.log("assetIdentifier not found in local photo library")
        }

        Self.log("falling back to NSItemProvider image loading")
        return try await loadSelectedPhotoFromProvider(result.itemProvider)
    }

    private func loadSelectedPhoto(from asset: PHAsset) async throws -> SelectedPhoto {
        Self.log("requestImage from asset, pixelSize=\(asset.pixelWidth)x\(asset.pixelHeight)")
        let requestOptions = PHImageRequestOptions()
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.resizeMode = .fast
        requestOptions.isNetworkAccessAllowed = true
        requestOptions.isSynchronous = false

        let longEdge = LocalVLMService.preferredPhotoRequestLongEdge(for: traitCollection.userInterfaceIdiom)
        let aspectRatio = CGFloat(asset.pixelWidth) / max(CGFloat(asset.pixelHeight), 1)
        let targetSize: CGSize
        if aspectRatio >= 1 {
            targetSize = CGSize(width: longEdge, height: round(longEdge / aspectRatio))
        } else {
            targetSize = CGSize(width: round(longEdge * aspectRatio), height: longEdge)
        }
        Self.log("requestImage targetSize=\(targetSize.width)x\(targetSize.height)")

        return try await withCheckedThrowingContinuation { continuation in
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFit,
                options: requestOptions
            ) { image, info in
                if let error = info?[PHImageErrorKey] as? Error {
                    Self.log("PHImageManager requestImage error: \(error.localizedDescription)")
                    continuation.resume(throwing: error)
                    return
                }
                if (info?[PHImageCancelledKey] as? Bool) == true {
                    Self.log("PHImageManager requestImage cancelled")
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true {
                    Self.log("PHImageManager returned degraded image, waiting for final image")
                    return
                }
                guard let image else {
                    Self.log("PHImageManager returned nil image")
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                Self.log("PHImageManager returned final image size=\(image.size.width)x\(image.size.height)")
                continuation.resume(
                    returning: SelectedPhoto(
                        image: image,
                        originalPixelSize: CGSize(width: asset.pixelWidth, height: asset.pixelHeight)
                    )
                )
            }
        }
    }

    private func loadSelectedPhotoFromProvider(_ provider: NSItemProvider) async throws -> SelectedPhoto {
        Self.log("loadSelectedPhotoFromProvider start")
        return try await withCheckedThrowingContinuation { continuation in
            provider.loadObject(ofClass: UIImage.self) { reading, error in
                if let error {
                    Self.log("NSItemProvider loadObject error: \(error.localizedDescription)")
                    continuation.resume(throwing: error)
                    return
                }
                guard let image = reading as? UIImage else {
                    Self.log("NSItemProvider returned non-UIImage object")
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                let pixelSize = image.size.applying(.init(scaleX: image.scale, y: image.scale))
                Self.log("NSItemProvider returned image size=\(pixelSize.width)x\(pixelSize.height)")
                continuation.resume(
                    returning: SelectedPhoto(
                        image: image,
                        originalPixelSize: pixelSize
                    )
                )
            }
        }
    }
}

final class ViewController: LocalVLMTestViewController {}

private extension LocalVLMTestViewController {
    static func log(_ message: String) {
        print("[LocalVLMTestViewController] \(message)")
    }
}
