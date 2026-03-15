import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import UIKit
import Vision

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

struct OCRResult: Sendable {
    let lines: [String]
    let fullText: String
    let preparationDuration: TimeInterval
    let recognitionDuration: TimeInterval
    let totalDuration: TimeInterval
}

enum LocalVLMError: LocalizedError {
    case missingModelDirectory
    case invalidSelectedImage
    case simulatorUnsupported

    var errorDescription: String? {
        switch self {
        case .missingModelDirectory:
            return "未找到 app bundle 中的 VL 模型文件。"
        case .invalidSelectedImage:
            return "无法读取所选照片。"
        case .simulatorUnsupported:
            return "iOS 模拟器不支持 MLX 本地 VL 推理，请用真机验证。"
        }
    }
}

actor LocalVLMService {
    static let shared = LocalVLMService()

    private static let phoneMaxInputLongEdge: CGFloat = 1024
    private static let padMaxInputLongEdge: CGFloat = 2016
    private static let promptResourceName = "vl_prompt"
    private static let memoryCacheLimitBytes = 32 * 1024 * 1024
    private static let ocrRecognitionLanguages = ["zh-Hans", "en-US"]
    private static let memoryScoreWeights: [String: Double] = [
        "mb": 0.18,
        "st": 0.15,
        "rl": 0.14,
        "em": 0.12,
        "an": 0.10,
        "ls": 0.10,
        "rv": 0.09,
        "wm": 0.07,
        "un": 0.05,
    ]
    private static let scoreKeyAliases: [String: [String]] = [
        "mb": ["mb", "memorability"],
        "em": ["em", "emotion"],
        "st": ["st", "story"],
        "an": ["an", "anchors"],
        "rl": ["rl", "relationship"],
        "ls": ["ls", "life_stage"],
        "un": ["un", "uniqueness"],
        "wm": ["wm", "warmth"],
        "rv": ["rv", "revisit_value"],
    ]
    private static let topLevelKeyAliases: [String: [String]] = [
        "ob": ["ob", "observation", "image_type", "sc", "scene"],
        "hh": ["hh", "has_human"],
        "ma": ["ma", "memorability_analysis", "dimension_scores"],
        "cl": ["cl", "caption_line"],
        "tg": ["tg", "tags"],
        "ti": ["ti", "text_in_image"],
        "ms": ["ms", "memory_score"],
        "oc": ["oc", "ocr"],
    ]
    private static let hasHumanWeight = 0.10
    private let modelFolderName = "Qwen3-VL-2B-Instruct-4bit"
    private var modelContainer: ModelContainer?
    private var modelContainerTask: Task<ModelContainer, Error>?
    private var activeSession: ChatSession?
    private var cancelledSession: ChatSession?
    private var isApplicationActive = true
    private let prompt: String

    private static let defaultPrompt = """
    请分析这张图片，并严格输出 JSON，不要输出额外说明。

    你需要完成 4 件事：

    1. 用一句简短中文描述图片可见内容，并让这句话适合后续搜索。
    这句话既要忠于画面，也要尽量点出可检索的对象或场景类别。
    例如如果明显是一张身份证、菜单、聊天截图、证件照、宠物照、旅行合影，就直接说出来。
    即使图片里没有这些文字，只要画面能判断出来，也要在描述里体现。

    字段：
    "ob": "一句适合搜索的简短中文描述"

    2. 评估这张图片的可回忆度。
    “可回忆度”指这张图片在未来是否容易唤起具体、鲜明、可讲述的个人回忆。
    它不是摄影质量分，也不是美观分。

    请从以下 4 个维度评分，每项 0-10 分：
    - mb：整体是否容易让人记住
    - em：是否有明显情绪或氛围
    - st：是否像一个可讲述的时刻
    - an：是否有具体细节可作为回忆锚点，如人物、动作、地点、物品、文字

    "ma": {
      "mb": 0,
      "em": 0,
      "st": 0,
      "an": 0
    }

    不要输出 ms，总分会由系统在你返回 ma 后自动计算，并补回最终结果。

    3. 给图片打 tag。
    请输出 3 到 8 个简短中文 tag。
    tag 要尽量具体，优先基于画面中真实可见的内容，不要空泛，不要为了好看而编造。

    字段：
    "tg": ["", "", ""]

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
    "cl": "一句题注"

    请严格输出 JSON，格式如下：
    {
      "ob": "",
      "ma": {
        "mb": 0,
        "em": 0,
        "st": 0,
        "an": 0
      },
      "tg": [],
      "cl": ""
    }
    """

    init() {
        #if !targetEnvironment(simulator)
        Memory.cacheLimit = Self.memoryCacheLimitBytes
        #endif
        self.prompt = Self.loadPrompt()
        Self.log("initialized, cacheLimit=\(Self.memoryCacheLimitBytes) bytes")
    }

    nonisolated static func preferredPhotoRequestLongEdge(for idiom: UIUserInterfaceIdiom) -> CGFloat {
        maxInputLongEdge(for: idiom)
    }

    func applicationDidBecomeActive() {
        guard !isApplicationActive else { return }
        isApplicationActive = true
        Self.log("applicationDidBecomeActive")
    }

    func applicationWillResignActive() async {
        guard isApplicationActive else { return }
        isApplicationActive = false
        Self.log("applicationWillResignActive")
        await cancelActiveRun(synchronizeCleanup: false)
    }

    func streamImage(
        _ imageData: Data,
        ocrImageData: Data? = nil,
        ocrPreparationDuration: TimeInterval = 0,
        userInterfaceIdiom: UIUserInterfaceIdiom,
        originalPixelSize: CGSize? = nil,
        onLoadProgress: @escaping @Sendable (Double) async -> Void,
        onImageInfo: @escaping @Sendable (ImageSizingInfo) async -> Void,
        onChunk: @escaping @Sendable (String) async -> Void,
        onComplete: @escaping @Sendable (VLMRunMetrics) async -> Void
    ) async throws -> String {
        #if targetEnvironment(simulator)
        throw LocalVLMError.simulatorUnsupported
        #else
        Self.log("streamImage start, imageDataBytes=\(imageData.count), idiom=\(userInterfaceIdiom.rawValue)")
        try ensureRunAllowed()
        let modelContainer = try await resolveModelContainer(onLoadProgress: onLoadProgress)
        try ensureRunAllowed()
        Self.log("model container ready")
        let preparedImage: (image: UserInput.Image, info: ImageSizingInfo) = try autoreleasepool {
            guard let image = UIImage(data: imageData) else {
                throw LocalVLMError.invalidSelectedImage
            }
            return Self.prepareInputImage(
                from: image,
                originalPixelSize: originalPixelSize,
                userInterfaceIdiom: userInterfaceIdiom
            )
        }
        try ensureRunAllowed()
        let session = ChatSession(modelContainer, processing: .init(resize: nil))
        activeSession = session
        cancelledSession = nil
        print(
            "VL input image size: original \(preparedImage.info.originalWidth)x\(preparedImage.info.originalHeight), "
                + "effective \(preparedImage.info.effectiveWidth)x\(preparedImage.info.effectiveHeight)"
        )
        Self.log(
            "prepared image, original=\(preparedImage.info.originalWidth)x\(preparedImage.info.originalHeight), "
                + "effective=\(preparedImage.info.effectiveWidth)x\(preparedImage.info.effectiveHeight)"
        )
        await onImageInfo(preparedImage.info)

        var output = ""
        do {
            for try await generation in session.streamDetails(
                to: prompt,
                images: [preparedImage.image],
                videos: []
            ) {
                try ensureRunAllowed()
                if let chunk = generation.chunk {
                    output += chunk
                    print(chunk, terminator: "")
                    await onChunk(chunk)
                }
                if let info = generation.info {
                    Self.log(
                        "generation info, promptTokens=\(info.promptTokenCount), generatedTokens=\(info.generationTokenCount), "
                            + "promptTime=\(info.promptTime), generateTime=\(info.generateTime), tps=\(info.tokensPerSecond)"
                    )
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
            if cancelledSession === session {
                cancelledSession = nil
                await finishActiveSession(session, synchronize: false)
                Self.log("streamImage cancelled after stream termination")
                throw CancellationError()
            }
            await finishActiveSession(session)
            Self.log("session cleared, MLX cache cleared")
            print("")
            let ocrResult: OCRResult?
            do {
                let ocrSourceData = ocrImageData ?? imageData
                ocrResult = try Self.recognizeText(in: ocrSourceData, preparationDuration: ocrPreparationDuration)
                if let ocrResult {
                    Self.log(
                        "ocr complete, sourceBytes=\(ocrSourceData.count), lines=\(ocrResult.lines.count), "
                            + "chars=\(ocrResult.fullText.count), preparationMs=\(Int((ocrResult.preparationDuration * 1000).rounded())), "
                            + "recognitionMs=\(Int((ocrResult.recognitionDuration * 1000).rounded())), "
                            + "totalMs=\(Int((ocrResult.totalDuration * 1000).rounded()))"
                    )
                    print("[LocalVLMService] OCR lines: \(ocrResult.lines)")
                    print("[LocalVLMService] OCR text:\n\(ocrResult.fullText)")
                    print(
                        "[LocalVLMService] OCR timing: preparation=\(Int((ocrResult.preparationDuration * 1000).rounded()))ms, "
                            + "recognition=\(Int((ocrResult.recognitionDuration * 1000).rounded()))ms, "
                            + "total=\(Int((ocrResult.totalDuration * 1000).rounded()))ms"
                    )
                }
            } catch {
                ocrResult = nil
                Self.log("ocr failed: \(error.localizedDescription)")
            }
            let finalOutput = Self.augmentedOutput(from: output, ocrResult: ocrResult)
            Self.log("streamImage completed, rawOutputChars=\(output.count), finalOutputChars=\(finalOutput.count)")
            return finalOutput
        } catch is CancellationError {
            if cancelledSession === session {
                cancelledSession = nil
            }
            await finishActiveSession(session, synchronize: false)
            Self.log("streamImage cancelled")
            throw CancellationError()
        } catch {
            if cancelledSession === session {
                cancelledSession = nil
            }
            await finishActiveSession(session)
            Self.log("streamImage failed: \(error.localizedDescription)")
            throw error
        }
        #endif
    }

    func cancelActiveRun() async {
        await cancelActiveRun(synchronizeCleanup: isApplicationActive)
    }

    private func cancelActiveRun(synchronizeCleanup: Bool) async {
        if let modelContainerTask {
            modelContainerTask.cancel()
            self.modelContainerTask = nil
            Self.log("cancelled in-flight model container task")
        }
        guard let activeSession else { return }
        cancelledSession = activeSession
        await activeSession.cancel()
        await finishActiveSession(activeSession, synchronize: synchronizeCleanup)
        Self.log("cancelActiveRun completed")
    }

    private func resolveModelContainer(
        onLoadProgress: @escaping @Sendable (Double) async -> Void
    ) async throws -> ModelContainer {
        if let modelContainer {
            Self.log("reusing cached model container")
            await onLoadProgress(1)
            return modelContainer
        }
        if let modelContainerTask {
            Self.log("awaiting in-flight model container task")
            let modelContainer = try await modelContainerTask.value
            await onLoadProgress(1)
            return modelContainer
        }

        let task = Task<ModelContainer, Error> {
            let modelURL = try Self.modelDirectoryURL(folderName: modelFolderName)
            Self.log("loading model from \(modelURL.path)")
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
            Self.log("model container load completed")
            await onLoadProgress(1)
            return modelContainer
        } catch {
            self.modelContainerTask = nil
            Self.log("model container load failed: \(error.localizedDescription)")
            throw error
        }
    }

    private func ensureRunAllowed() throws {
        try Task.checkCancellation()
        guard isApplicationActive else {
            Self.log("rejecting VLM work while application is inactive")
            throw CancellationError()
        }
    }

    private func finishActiveSession(_ session: ChatSession, synchronize: Bool? = nil) async {
        await session.clear()
        Memory.clearCache()
        if synchronize ?? isApplicationActive {
            Stream().synchronize()
        }
        if activeSession === session {
            activeSession = nil
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
                Self.log("resolved model directory at \(directory.path)")
                return directory
            }
        }

        Self.log("failed to resolve model directory for folder \(folderName)")
        throw LocalVLMError.missingModelDirectory
    }

    private static func loadPrompt() -> String {
        guard let promptURL = Bundle.main.url(forResource: promptResourceName, withExtension: "md") else {
            print("VL prompt resource not found, using embedded default prompt.")
            log("prompt resource missing, using embedded default")
            return defaultPrompt
        }

        do {
            let prompt = try String(contentsOf: promptURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !prompt.isEmpty else {
                print("VL prompt resource is empty, using embedded default prompt.")
                log("prompt resource empty at \(promptURL.path), using embedded default")
                return defaultPrompt
            }
            log("loaded prompt from \(promptURL.path), chars=\(prompt.count)")
            return prompt
        } catch {
            print("Failed to load VL prompt resource: \(error.localizedDescription). Using embedded default prompt.")
            log("failed to load prompt: \(error.localizedDescription), using embedded default")
            return defaultPrompt
        }
    }

    private static func augmentedOutput(from rawOutput: String, ocrResult: OCRResult?) -> String {
        log("augment start, rawOutputChars=\(rawOutput.count)")
        guard
            let cleanedOutput = cleanedJSONString(from: rawOutput),
            let jsonData = cleanedOutput.data(using: .utf8),
            let jsonObject = try? JSONSerialization.jsonObject(with: jsonData),
            var root = jsonObject as? [String: Any]
        else {
            log("augment skipped, unable to parse JSON from output")
            return rawOutput
        }
        log("cleaned JSON chars=\(cleanedOutput.count)")

        let analysisKey: String
        if root["ma"] is [String: Any] {
            analysisKey = "ma"
        } else if root["memorability_analysis"] is [String: Any] {
            analysisKey = "memorability_analysis"
        } else if root["dimension_scores"] is [String: Any] {
            analysisKey = "dimension_scores"
        } else {
            log("augment skipped, no ma or memorability_analysis or dimension_scores key")
            return rawOutput
        }
        log("using analysisKey=\(analysisKey)")

        guard let rawScores = root[analysisKey] as? [String: Any] else {
            log("augment skipped, analysis payload is not dictionary")
            return rawOutput
        }
        var scores = rawScores

        var weightedScore = 0.0
        var totalWeight = 0.0
        var normalizedScores: [String: Any] = [:]

        for (key, weight) in memoryScoreWeights {
            let value = normalizedScoreValue(from: rawScoreValue(for: key, in: scores))
            normalizedScores[key] = value
            weightedScore += value * weight
            totalWeight += weight
            log("score[\(key)]=\(value), weight=\(weight)")
        }

        for (key, value) in scores {
            guard normalizedScores[key] == nil else { continue }
            guard !scoreKeyAliases.values.flatMap({ $0 }).contains(key) else { continue }
            normalizedScores[key] = normalizedScoreValue(from: value)
        }

        root["ma"] = normalizedScores
        root.removeValue(forKey: "memorability_analysis")
        root.removeValue(forKey: "dimension_scores")

        let rawHasHuman = rawTopLevelValue(for: "hh", in: root)
        if let hasHuman = rawHasHuman as? Bool {
            let humanScore = hasHuman ? 8.0 : 4.0
            weightedScore += humanScore * hasHumanWeight
            totalWeight += hasHumanWeight
            root["hh"] = hasHuman
            log("hh(bool)=\(hasHuman), mappedScore=\(humanScore), weight=\(hasHumanWeight)")
        } else if let hasHumanNumber = rawHasHuman as? NSNumber {
            let humanScore = hasHumanNumber.boolValue ? 8.0 : 4.0
            weightedScore += humanScore * hasHumanWeight
            totalWeight += hasHumanWeight
            root["hh"] = hasHumanNumber.boolValue
            log("hh(number)=\(hasHumanNumber), mappedScore=\(humanScore), weight=\(hasHumanWeight)")
        } else if let hasHumanString = rawHasHuman as? String {
            let normalized = ["true", "1", "yes"].contains(hasHumanString.lowercased())
            let humanScore = normalized ? 8.0 : 4.0
            weightedScore += humanScore * hasHumanWeight
            totalWeight += hasHumanWeight
            root["hh"] = normalized
            log("hh(string)=\(hasHumanString), normalized=\(normalized), mappedScore=\(humanScore), weight=\(hasHumanWeight)")
        } else {
            log("hh missing, no additional weight applied")
        }

        guard totalWeight > 0 else {
            log("augment skipped, totalWeight=0")
            return rawOutput
        }

        let roundedMemoryScore = round((weightedScore / totalWeight) * 10) / 10
        root["ms"] = roundedMemoryScore
        log("computed memory_score=\(roundedMemoryScore), weightedScore=\(weightedScore), totalWeight=\(totalWeight)")
        mergeOCRResult(ocrResult, into: &root)
        normalizeTopLevelKeys(in: &root)

        guard
            let normalizedData = try? JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted]),
            let normalizedString = String(data: normalizedData, encoding: .utf8)
        else {
            log("augment failed, unable to serialize normalized JSON")
            return rawOutput
        }

        log("augment complete, finalJSONChars=\(normalizedString.count)")
        return normalizedString
    }

    private static func mergeOCRResult(_ ocrResult: OCRResult?, into root: inout [String: Any]) {
        let lines = ocrResult?.lines ?? []
        let fullText = ocrResult?.fullText ?? ""
        let durationMs = Int(((ocrResult?.totalDuration ?? 0) * 1000).rounded())
        let preparationMs = Int(((ocrResult?.preparationDuration ?? 0) * 1000).rounded())
        let recognitionMs = Int(((ocrResult?.recognitionDuration ?? 0) * 1000).rounded())

        root["ti"] = lines
        root["oc"] = [
            "tx": fullText,
            "lc": lines.count,
            "dm": durationMs,
            "pm": preparationMs,
            "rm": recognitionMs,
            "en": "vision_vnrecognizetextrequest",
        ]
        log(
            "merged OCR result, lines=\(lines.count), chars=\(fullText.count), "
                + "preparationMs=\(preparationMs), recognitionMs=\(recognitionMs), totalMs=\(durationMs)"
        )
    }

    private static func recognizeText(in imageData: Data, preparationDuration: TimeInterval) throws -> OCRResult {
        let startedAt = Date()
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ocrRecognitionLanguages
        request.minimumTextHeight = 0

        let handler = VNImageRequestHandler(data: imageData, options: [:])
        try handler.perform([request])

        let observations = request.results ?? []
        var lines: [String] = []
        var seen = Set<String>()

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let candidateLines = candidate.string
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            for line in candidateLines where seen.insert(line).inserted {
                lines.append(line)
            }
        }

        let fullText = lines.joined(separator: "\n")
        let recognitionDuration = Date().timeIntervalSince(startedAt)
        return OCRResult(
            lines: lines,
            fullText: fullText,
            preparationDuration: preparationDuration,
            recognitionDuration: recognitionDuration,
            totalDuration: preparationDuration + recognitionDuration
        )
    }

    private static func cleanedJSONString(from rawOutput: String) -> String? {
        let trimmed = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            log("cleanedJSONString: output empty after trim")
            return nil
        }

        if trimmed.hasPrefix("```") {
            let lines = trimmed.components(separatedBy: .newlines)
            guard lines.count >= 3 else {
                log("cleanedJSONString: fenced block too short")
                return nil
            }

            let bodyLines = Array(lines.dropFirst().dropLast())
            let fencedBody = bodyLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !fencedBody.isEmpty {
                log("cleanedJSONString: stripped fenced code block")
                return fencedBody
            }
        }

        log("cleanedJSONString: using trimmed raw output")
        return trimmed
    }

    private static func normalizedScoreValue(from rawValue: Any?) -> Double {
        let numericValue: Double
        switch rawValue {
        case let value as NSNumber:
            numericValue = value.doubleValue
        case let value as String:
            numericValue = Double(value) ?? 0
        default:
            numericValue = 0
        }

        let clamped = min(max(numericValue, 0), 10)
        let rounded = round(clamped * 10) / 10
        if rawValue != nil {
            log("normalized score from \(String(describing: rawValue)) -> \(rounded)")
        }
        return rounded
    }

    private static func rawScoreValue(for key: String, in scores: [String: Any]) -> Any? {
        for alias in scoreKeyAliases[key] ?? [key] {
            if let value = scores[alias] {
                if alias != key {
                    log("mapped score alias \(alias) -> \(key)")
                }
                return value
            }
        }
        return nil
    }

    private static func rawTopLevelValue(for key: String, in root: [String: Any]) -> Any? {
        for alias in topLevelKeyAliases[key] ?? [key] {
            if let value = root[alias] {
                if alias != key {
                    log("mapped top-level alias \(alias) -> \(key)")
                }
                return value
            }
        }
        return nil
    }

    private static func normalizeTopLevelKeys(in root: inout [String: Any]) {
        if let observationDescription = normalizedObservationDescription(from: root) {
            root["ob"] = observationDescription
        }

        for (shortKey, aliases) in topLevelKeyAliases {
            guard let value = rawTopLevelValue(for: shortKey, in: root) else { continue }
            root[shortKey] = value
            for alias in aliases where alias != shortKey {
                root.removeValue(forKey: alias)
            }
        }

        root.removeValue(forKey: "mr")
        root.removeValue(forKey: "memory_reason")
    }

    private static func normalizedObservationDescription(from root: [String: Any]) -> String? {
        let observation = stringValue(from: root["ob"])
            ?? stringValue(from: root["observation"])
            ?? stringValue(from: root["image_type"])
        let scene = stringValue(from: root["sc"]) ?? stringValue(from: root["scene"])

        switch (observation, scene) {
        case let (.some(observation), .some(scene)):
            if observation.contains(scene) || scene.contains(observation) {
                return observation.count >= scene.count ? observation : scene
            }
            return "\(observation)，\(scene)"
        case let (.some(observation), nil):
            return observation
        case let (nil, .some(scene)):
            return scene
        default:
            return nil
        }
    }

    private static func stringValue(from rawValue: Any?) -> String? {
        guard let rawValue else { return nil }
        let string: String
        switch rawValue {
        case let stringValue as String:
            string = stringValue
        case let number as NSNumber:
            string = number.stringValue
        default:
            return nil
        }

        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
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

    private static func log(_ message: String) {
        print("[LocalVLMService] \(message)")
    }
}
