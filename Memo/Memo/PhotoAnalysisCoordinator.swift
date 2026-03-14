import Photos
import UIKit

struct PhotoAnalysisProgressSnapshot {
    let processedCount: Int
    let totalCount: Int
    let isVisible: Bool
    let etaText: String?
}

@MainActor
final class PhotoAnalysisCoordinator {
    static let shared = PhotoAnalysisCoordinator()

    private static let defaultEstimatedSecondsPerItem: TimeInterval = 18

    private let libraryService = PhotoLibraryService.shared
    private let store = PhotoAnalysisStore()

    private var itemStates: [String: PhotoAnalysisStatus] = [:]
    private var pendingAssetIdentifiers: [String] = []
    private var runningAssetIdentifier: String?
    private var workerTask: Task<Void, Never>?
    private var rebuildTask: Task<Void, Never>?
    private var observer: NSObjectProtocol?
    private var started = false
    private var processingEnabled = false
    private var backgroundTaskIdentifier: UIBackgroundTaskIdentifier = .invalid
    private var estimatedSecondsPerItem: TimeInterval = defaultEstimatedSecondsPerItem

    private init() {}

    deinit {
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    func start() {
        guard !started else { return }
        started = true

        observer = NotificationCenter.default.addObserver(
            forName: .photoLibraryServiceDidUpdate,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.scheduleRebuild()
            }
        }

        scheduleRebuild()
    }

    func sceneDidBecomeActive() {
        finishBackgroundTaskIfNeeded()
        start()
        scheduleRebuild()
    }

    func sceneDidEnterBackground() {
        guard runningAssetIdentifier != nil || !pendingAssetIdentifiers.isEmpty else {
            finishBackgroundTaskIfNeeded()
            return
        }
        beginBackgroundTaskIfNeeded()
    }

    func progressSnapshot() -> PhotoAnalysisProgressSnapshot {
        let totalCount = libraryService.currentAssets().count
        let processedCount = itemStates.values.filter { $0 == .completed || $0 == .failed }.count
        let remainingCount = max(totalCount - processedCount, 0)
        let isVisible = processingEnabled && totalCount > 0 && (runningAssetIdentifier != nil || !pendingAssetIdentifiers.isEmpty)
        return PhotoAnalysisProgressSnapshot(
            processedCount: processedCount,
            totalCount: totalCount,
            isVisible: isVisible,
            etaText: isVisible ? Self.bucketedETA(for: remainingCount, secondsPerItem: estimatedSecondsPerItem) : nil
        )
    }

    func status(for assetLocalIdentifier: String) -> PhotoAnalysisStatus {
        itemStates[assetLocalIdentifier] ?? .pending
    }

    func setProcessingEnabled(_ enabled: Bool) {
        guard processingEnabled != enabled else { return }
        processingEnabled = enabled
        print("[PhotoAnalysisCoordinator] processingEnabled=\(enabled)")

        if enabled {
            start()
            scheduleRebuild()
        } else {
            finishBackgroundTaskIfNeeded()
            notifyDidUpdate()
        }
    }

    private func scheduleRebuild() {
        rebuildTask?.cancel()
        rebuildTask = Task { [weak self] in
            guard let self else { return }
            await self.rebuildState()
        }
    }

    private func rebuildState() async {
        let assets = libraryService.currentAssets()
        let records = await store.loadAllRecords()
        estimatedSecondsPerItem = Self.estimatedDuration(from: Array(records.values))

        var nextStates: [String: PhotoAnalysisStatus] = [:]
        var nextPending: [String] = []

        for asset in assets {
            let identifier = asset.localIdentifier

            if identifier == runningAssetIdentifier {
                nextStates[identifier] = .running
                continue
            }

            if let record = records[identifier], Self.record(record, matches: asset) {
                nextStates[identifier] = record.status
            } else {
                nextStates[identifier] = .pending
                nextPending.append(identifier)
            }
        }

        itemStates = nextStates
        pendingAssetIdentifiers = nextPending

        if let runningAssetIdentifier, nextStates[runningAssetIdentifier] == nil {
            self.runningAssetIdentifier = nil
        }

        notifyDidUpdate()
        startProcessingIfNeeded()
    }

    private func startProcessingIfNeeded() {
        guard workerTask == nil else { return }
        guard processingEnabled else {
            notifyDidUpdate()
            return
        }
        guard libraryService.hasFullAccess else {
            notifyDidUpdate()
            return
        }
        guard !pendingAssetIdentifiers.isEmpty else {
            finishBackgroundTaskIfNeeded()
            notifyDidUpdate()
            return
        }

        beginBackgroundTaskIfNeeded()
        workerTask = Task { [weak self] in
            guard let self else { return }
            await self.processQueue()
        }
    }

    private func processQueue() async {
        while let nextIdentifier = nextPendingAssetIdentifier() {
            do {
                try await analyzeAsset(withLocalIdentifier: nextIdentifier)
            } catch {
                print("[PhotoAnalysisCoordinator] analyze failed for \(nextIdentifier): \(error.localizedDescription)")
                await markFailure(for: nextIdentifier, errorMessage: error.localizedDescription)
            }
        }

        workerTask = nil
        finishBackgroundTaskIfNeeded()
        notifyDidUpdate()
    }

    private func nextPendingAssetIdentifier() -> String? {
        guard !pendingAssetIdentifiers.isEmpty else {
            return nil
        }

        let identifier = pendingAssetIdentifiers.removeFirst()
        runningAssetIdentifier = identifier
        itemStates[identifier] = .running
        notifyDidUpdate()
        return identifier
    }

    private func analyzeAsset(withLocalIdentifier identifier: String) async throws {
        guard let asset = libraryService.asset(withLocalIdentifier: identifier) else {
            runningAssetIdentifier = nil
            itemStates.removeValue(forKey: identifier)
            notifyDidUpdate()
            return
        }

        let analysisStartedAt = Date()
        let selectedPhoto = try await libraryService.requestAnalysisPhoto(
            for: asset,
            userInterfaceIdiom: UIDevice.current.userInterfaceIdiom
        )
        let imageData = selectedPhoto.image.normalizedJPEGData()

        let resultJSONString = try await LocalVLMService.shared.streamImage(
            imageData,
            userInterfaceIdiom: UIDevice.current.userInterfaceIdiom,
            originalPixelSize: selectedPhoto.originalPixelSize,
            onLoadProgress: { _ in },
            onImageInfo: { _ in },
            onChunk: { _ in },
            onComplete: { _ in }
        )
        let analysisDuration = Date().timeIntervalSince(analysisStartedAt)

        let record = PhotoAnalysisRecord(
            assetLocalIdentifier: identifier,
            assetCreationTimestamp: asset.creationDate?.timeIntervalSince1970,
            assetModificationTimestamp: asset.modificationDate?.timeIntervalSince1970,
            status: .completed,
            updatedAt: Date().timeIntervalSince1970,
            analysisDurationSeconds: analysisDuration,
            memoryScore: Self.extractMemoryScore(from: resultJSONString),
            resultJSONString: resultJSONString,
            errorMessage: nil
        )

        await store.save(record)

        runningAssetIdentifier = nil
        itemStates[identifier] = .completed
        estimatedSecondsPerItem = Self.updatedEstimatedDuration(current: estimatedSecondsPerItem, sample: analysisDuration)
        print("[PhotoAnalysisCoordinator] analyzed \(identifier)")
        notifyDidUpdate()
    }

    private func markFailure(for identifier: String, errorMessage: String) async {
        let asset = libraryService.asset(withLocalIdentifier: identifier)
        let record = PhotoAnalysisRecord(
            assetLocalIdentifier: identifier,
            assetCreationTimestamp: asset?.creationDate?.timeIntervalSince1970,
            assetModificationTimestamp: asset?.modificationDate?.timeIntervalSince1970,
            status: .failed,
            updatedAt: Date().timeIntervalSince1970,
            analysisDurationSeconds: nil,
            memoryScore: nil,
            resultJSONString: nil,
            errorMessage: errorMessage
        )

        await store.save(record)
        runningAssetIdentifier = nil
        itemStates[identifier] = .failed
        notifyDidUpdate()
    }

    private func notifyDidUpdate() {
        NotificationCenter.default.post(name: .photoAnalysisCoordinatorDidUpdate, object: nil)
    }

    private func beginBackgroundTaskIfNeeded() {
        guard backgroundTaskIdentifier == .invalid else { return }
        backgroundTaskIdentifier = UIApplication.shared.beginBackgroundTask(withName: "MemoPhotoAnalysis") { [weak self] in
            Task { @MainActor in
                self?.finishBackgroundTaskIfNeeded()
            }
        }
    }

    private func finishBackgroundTaskIfNeeded() {
        guard backgroundTaskIdentifier != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTaskIdentifier)
        backgroundTaskIdentifier = .invalid
    }

    private static func record(_ record: PhotoAnalysisRecord, matches asset: PHAsset) -> Bool {
        let recordModification = record.assetModificationTimestamp ?? 0
        let assetModification = asset.modificationDate?.timeIntervalSince1970 ?? 0
        return abs(recordModification - assetModification) < 1
    }

    private static func extractMemoryScore(from jsonString: String) -> Double? {
        guard
            let data = jsonString.data(using: .utf8),
            let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        if let number = jsonObject["memory_score"] as? NSNumber {
            return number.doubleValue
        }

        if let string = jsonObject["memory_score"] as? String {
            return Double(string)
        }

        return nil
    }

    private static func estimatedDuration(from records: [PhotoAnalysisRecord]) -> TimeInterval {
        let durations = records.compactMap { record -> TimeInterval? in
            guard record.status == .completed else { return nil }
            guard let duration = record.analysisDurationSeconds, duration > 0 else { return nil }
            return duration
        }

        guard !durations.isEmpty else {
            return defaultEstimatedSecondsPerItem
        }

        let average = durations.reduce(0, +) / Double(durations.count)
        return max(6, min(180, average))
    }

    private static func updatedEstimatedDuration(current: TimeInterval, sample: TimeInterval) -> TimeInterval {
        let clampedSample = max(6, min(180, sample))
        return current * 0.7 + clampedSample * 0.3
    }

    private static func bucketedETA(for remainingCount: Int, secondsPerItem: TimeInterval) -> String? {
        guard remainingCount > 0 else { return nil }
        let remainingSeconds = Double(remainingCount) * secondsPerItem
        if remainingSeconds < 45 {
            return "预计不到 1 分钟"
        }
        if remainingSeconds < 90 {
            return "预计约 1 分钟"
        }
        if remainingSeconds < 10 * 60 {
            let minutes = Int((remainingSeconds / 60).rounded())
            return "预计约 \(minutes) 分钟"
        }
        if remainingSeconds < 60 * 60 {
            let minutes = Int((remainingSeconds / 300).rounded()) * 5
            return "预计约 \(max(minutes, 10)) 分钟"
        }
        if remainingSeconds < 4 * 60 * 60 {
            let halfHours = (remainingSeconds / 1800).rounded()
            let hours = halfHours / 2
            if halfHours.truncatingRemainder(dividingBy: 2) == 0 {
                return "预计约 \(Int(hours)) 小时"
            }
            return "预计约 \(Int(floor(hours))) 小时半"
        }
        let hours = Int((remainingSeconds / 3600).rounded())
        return "预计约 \(max(hours, 4)) 小时"
    }
}
