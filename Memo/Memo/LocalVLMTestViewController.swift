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
    private var selectedImage: UIImage?
    private var selectedImageData: Data?
    private var selectedOriginalPixelSize: CGSize?

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "VL 测试"
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
        Task { [weak self] in
            guard let self else { return }
            let granted = await self.ensureFullPhotoLibraryAccess()
            guard granted else { return }
            await MainActor.run {
                var configuration = PHPickerConfiguration(photoLibrary: .shared())
                configuration.filter = .images
                configuration.selectionLimit = 1

                let picker = PHPickerViewController(configuration: configuration)
                picker.delegate = self
                self.present(picker, animated: true)
            }
        }
    }

    @objc
    private func runVLTest() {
        guard let selectedImageData else {
            statusLabel.text = "请先从系统相册选择一张照片。"
            return
        }
        hasReceivedChunk = false
        latestMetrics = nil
        runStartedAt = Date()
        imageView.isUserInteractionEnabled = false
        runButton.isEnabled = false
        spinner.startAnimating()
        statusLabel.text = "正在加载 2B VL 模型并流式生成..."
        outputView.text = "生成中..."

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
                    self.outputView.text = result
                    if let metrics = self.latestMetrics {
                        self.statusLabel.text = "生成完成。输出 \(metrics.generatedTokenCount) tokens，耗时 \(self.formatSeconds(metrics.totalTime))，\(self.formatRate(metrics.tokensPerSecond)) tok/s"
                    } else {
                        self.statusLabel.text = "流式生成完成。"
                    }
                    self.imageView.isUserInteractionEnabled = true
                    self.runButton.isEnabled = true
                    self.spinner.stopAnimating()
                }
            } catch {
                await MainActor.run {
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
        switch currentStatus {
        case .authorized:
            return true
        case .notDetermined:
            let newStatus = await requestPhotoLibraryAuthorization()
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
        picker.dismiss(animated: true)

        guard let result = results.first else {
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
                    self.selectedImage = selectedPhoto.image
                    self.selectedImageData = selectedPhoto.imageData
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
                    self.selectedImage = nil
                    self.selectedImageData = nil
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
            let assets = PHAsset.fetchAssets(withLocalIdentifiers: [assetIdentifier], options: nil)
            if let asset = assets.firstObject {
                return try await loadSelectedPhoto(from: asset)
            }
        }

        return try await loadSelectedPhotoFromProvider(result.itemProvider)
    }

    private func loadSelectedPhoto(from asset: PHAsset) async throws -> SelectedPhoto {
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

        return try await withCheckedThrowingContinuation { continuation in
            PHImageManager.default().requestImage(
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
                        imageData: image.normalizedJPEGData(),
                        originalPixelSize: CGSize(width: asset.pixelWidth, height: asset.pixelHeight)
                    )
                )
            }
        }
    }

    private func loadSelectedPhotoFromProvider(_ provider: NSItemProvider) async throws -> SelectedPhoto {
        return try await withCheckedThrowingContinuation { continuation in
            provider.loadObject(ofClass: UIImage.self) { reading, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let image = reading as? UIImage else {
                    continuation.resume(throwing: LocalVLMError.invalidSelectedImage)
                    return
                }
                let pixelSize = image.size.applying(.init(scaleX: image.scale, y: image.scale))
                continuation.resume(
                    returning: SelectedPhoto(
                        image: image,
                        imageData: image.normalizedJPEGData(),
                        originalPixelSize: pixelSize
                    )
                )
            }
        }
    }
}

private struct SelectedPhoto {
    let image: UIImage
    let imageData: Data
    let originalPixelSize: CGSize
}

private extension UIImage {
    func normalizedJPEGData() -> Data {
        if let data = jpegData(compressionQuality: 0.92) {
            return data
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let renderedImage = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
        return renderedImage.jpegData(compressionQuality: 0.92) ?? Data()
    }
}

final class ViewController: LocalVLMTestViewController {}
