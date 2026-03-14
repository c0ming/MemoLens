import Foundation
import MLXLMCommon
import MLXVLM
import UIKit

struct ImageSizingInfo: Sendable {
    let originalWidth: Int
    let originalHeight: Int
    let effectiveWidth: Int
    let effectiveHeight: Int
}

struct VLMRunMetrics: Sendable {
    let promptTokenCount: Int
    let generatedTokenCount: Int
    let promptTime: TimeInterval
    let generationTime: TimeInterval
    let tokensPerSecond: Double
    let totalTime: TimeInterval
}

enum LocalVLMError: LocalizedError {
    case missingModelDirectory
    case invalidSelectedImage

    var errorDescription: String? {
        switch self {
        case .missingModelDirectory:
            return "未找到 app bundle 中的 2B VL 模型文件。"
        case .invalidSelectedImage:
            return "无法读取所选照片。"
        }
    }
}

actor LocalVLMService {
    static let shared = LocalVLMService()

    private static let phoneMaxInputLongEdge: CGFloat = 1024
    private static let padMaxInputLongEdge: CGFloat = 2016
    private let modelFolderName = "Qwen3-VL-2B-Instruct-4bit"
    private var modelContainer: ModelContainer?
    private var modelContainerTask: Task<ModelContainer, Error>?
    private let prompt = """
    请分析这张图片，并严格输出 JSON，不要输出额外说明。

    你需要完成 4 件事：

    1. 用一句简短中文说明这是什么图片。
    不要套用固定类别，如果无法准确概括，就尽量如实描述。

    字段：
    "image_type": "一句简短中文"

    2. 评估这张图片的可回忆度。
    “可回忆度”指这张图片在未来是否容易唤起具体、鲜明、可讲述的个人回忆。
    它不是摄影质量分，也不是美观分。

    请从以下 4 个维度评分，每项 0-10 分：
    - memorability：整体是否容易让人记住
    - emotion：是否有明显情绪或氛围
    - story：是否像一个可讲述的时刻
    - anchors：是否有具体细节可作为回忆锚点，如人物、动作、地点、物品、文字

    再用一句话解释原因。

    字段：
    "dimension_scores": {
      "memorability": 0,
      "emotion": 0,
      "story": 0,
      "anchors": 0
    },
    "memory_reason": "一句话说明为什么这个分数高或低"

    不要输出 memory_score，总分会由系统在你返回 dimension_scores 后自动计算，并补回最终结果。

    3. 给图片打 tag。
    请输出 3 到 8 个简短中文 tag。
    tag 要尽量具体，优先基于画面中真实可见的内容，不要空泛，不要为了好看而编造。

    字段：
    "tags": ["", "", ""]

    4. 给图片写一句题注。
    要求：
    - 只写一句
    - 自然，不生硬
    - 可以有一点诗意，但不要过度
    - 不要像说明文
    - 不要空泛，不要鸡汤
    - 不要编造图片中看不出来的背景故事
    - 不要用“这张照片里……”“画面中……”这类开头

    字段：
    "caption_line": "一句题注"

    请严格输出 JSON，格式如下：
    {
      "image_type": "",
      "dimension_scores": {
        "memorability": 0,
        "emotion": 0,
        "story": 0,
        "anchors": 0
      },
      "memory_reason": "",
      "tags": [],
      "caption_line": ""
    }
    """

    nonisolated static func preferredPhotoRequestLongEdge(for idiom: UIUserInterfaceIdiom) -> CGFloat {
        maxInputLongEdge(for: idiom)
    }

    func streamImage(
        _ imageData: Data,
        userInterfaceIdiom: UIUserInterfaceIdiom,
        originalPixelSize: CGSize? = nil,
        onLoadProgress: @escaping @Sendable (Double) async -> Void,
        onImageInfo: @escaping @Sendable (ImageSizingInfo) async -> Void,
        onChunk: @escaping @Sendable (String) async -> Void,
        onComplete: @escaping @Sendable (VLMRunMetrics) async -> Void
    ) async throws -> String {
        let modelContainer = try await resolveModelContainer(onLoadProgress: onLoadProgress)
        guard let image = UIImage(data: imageData) else {
            throw LocalVLMError.invalidSelectedImage
        }
        let preparedImage = Self.prepareInputImage(
            from: image,
            originalPixelSize: originalPixelSize,
            userInterfaceIdiom: userInterfaceIdiom
        )
        let session = ChatSession(modelContainer, processing: .init(resize: nil))
        print(
            "VL input image size: original \(preparedImage.info.originalWidth)x\(preparedImage.info.originalHeight), "
                + "effective \(preparedImage.info.effectiveWidth)x\(preparedImage.info.effectiveHeight)"
        )
        await onImageInfo(preparedImage.info)

        var output = ""
        for try await generation in session.streamDetails(
            to: prompt,
            images: [preparedImage.image],
            videos: []
        ) {
            if let chunk = generation.chunk {
                output += chunk
                print(chunk, terminator: "")
                await onChunk(chunk)
            }
            if let info = generation.info {
                await onComplete(
                    VLMRunMetrics(
                        promptTokenCount: info.promptTokenCount,
                        generatedTokenCount: info.generationTokenCount,
                        promptTime: info.promptTime,
                        generationTime: info.generateTime,
                        tokensPerSecond: info.tokensPerSecond,
                        totalTime: info.promptTime + info.generateTime
                    )
                )
            }
        }
        print("")
        return output
    }

    private func resolveModelContainer(
        onLoadProgress: @escaping @Sendable (Double) async -> Void
    ) async throws -> ModelContainer {
        if let modelContainer {
            await onLoadProgress(1)
            return modelContainer
        }
        if let modelContainerTask {
            let modelContainer = try await modelContainerTask.value
            await onLoadProgress(1)
            return modelContainer
        }

        let task = Task<ModelContainer, Error> {
            let modelURL = try Self.modelDirectoryURL(folderName: modelFolderName)
            let configuration = ModelConfiguration(directory: modelURL)
            return try await VLMModelFactory.shared.loadContainer(configuration: configuration) { progress in
                print("VL model load progress: \(Int(progress.fractionCompleted * 100))%")
                Task {
                    await onLoadProgress(progress.fractionCompleted)
                }
            }
        }

        modelContainerTask = task

        do {
            let modelContainer = try await task.value
            self.modelContainer = modelContainer
            self.modelContainerTask = nil
            await onLoadProgress(1)
            return modelContainer
        } catch {
            self.modelContainerTask = nil
            throw error
        }
    }

    private static func modelDirectoryURL(folderName: String) throws -> URL {
        guard let resourceURL = Bundle.main.resourceURL else {
            throw LocalVLMError.missingModelDirectory
        }

        let candidateDirectories = [
            resourceURL.appendingPathComponent("LocalModels", isDirectory: true)
                .appendingPathComponent(folderName, isDirectory: true),
            resourceURL,
        ]

        for directory in candidateDirectories {
            let configURL = directory.appendingPathComponent("config.json")
            let weightsURL = directory.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: configURL.path),
               FileManager.default.fileExists(atPath: weightsURL.path) {
                return directory
            }
        }

        throw LocalVLMError.missingModelDirectory
    }

    private static func prepareInputImage(
        from image: UIImage,
        originalPixelSize: CGSize? = nil,
        userInterfaceIdiom: UIUserInterfaceIdiom
    ) -> (image: UserInput.Image, info: ImageSizingInfo) {
        let resolvedOriginalPixelSize = originalPixelSize ?? image.size.applying(.init(scaleX: image.scale, y: image.scale))
        let effectivePixelSize = scaledPixelSize(
            for: resolvedOriginalPixelSize,
            maxLongEdge: maxInputLongEdge(for: userInterfaceIdiom)
        )

        let info = ImageSizingInfo(
            originalWidth: Int(resolvedOriginalPixelSize.width.rounded()),
            originalHeight: Int(resolvedOriginalPixelSize.height.rounded()),
            effectiveWidth: Int(effectivePixelSize.width.rounded()),
            effectiveHeight: Int(effectivePixelSize.height.rounded())
        )

        guard effectivePixelSize != image.size else {
            if let ciImage = image.ciImage {
                return (.ciImage(ciImage), info)
            }
            if let cgImage = image.cgImage {
                return (.ciImage(CIImage(cgImage: cgImage)), info)
            }
            let format = UIGraphicsImageRendererFormat.default()
            format.scale = 1
            let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
            let renderedImage = renderer.image { _ in
                image.draw(in: CGRect(origin: .zero, size: image.size))
            }
            guard let renderedCGImage = renderedImage.cgImage else {
                let fallbackImage = CIImage(color: CIColor.black).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
                return (.ciImage(fallbackImage), info)
            }
            return (.ciImage(CIImage(cgImage: renderedCGImage)), info)
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: effectivePixelSize, format: format)
        let resizedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: effectivePixelSize))
        }

        guard let cgImage = resizedImage.cgImage else {
            let fallbackImage = CIImage(color: CIColor.black).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
            return (.ciImage(fallbackImage), info)
        }

        return (.ciImage(CIImage(cgImage: cgImage)), info)
    }

    private static func maxInputLongEdge(for idiom: UIUserInterfaceIdiom) -> CGFloat {
        switch idiom {
        case .pad:
            return padMaxInputLongEdge
        default:
            return phoneMaxInputLongEdge
        }
    }

    private static func scaledPixelSize(for size: CGSize, maxLongEdge: CGFloat) -> CGSize {
        guard size.width > 0, size.height > 0 else {
            return .zero
        }

        let longEdge = max(size.width, size.height)
        guard longEdge > maxLongEdge else {
            return size
        }

        let scale = maxLongEdge / longEdge
        return CGSize(
            width: round(size.width * scale),
            height: round(size.height * scale)
        )
    }
}
