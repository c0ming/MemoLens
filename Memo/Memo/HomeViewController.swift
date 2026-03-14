import Photos
import SnapKit
import UIKit

final class HomeViewController: UIViewController {
    private enum Layout {
        static let spacing: CGFloat = 12
        static let progressExpandedHeight: CGFloat = 84
        static let progressAnimationDuration: TimeInterval = 0.28
        static let progressAnimationOffset: CGFloat = 10
        static let analysisStartDelayNanoseconds: UInt64 = 3_000_000_000
    }

    private let scrollContainerView = UIView()
    private let progressContainerView = UIView()
    private let progressTitleLabel = UILabel()
    private let progressValueLabel = UILabel()
    private let progressEtaLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .default)
    private let unauthorizedContainerView = UIView()
    private let unauthorizedTitleLabel = UILabel()
    private let unauthorizedDetailLabel = UILabel()
    private let authorizeButton = UIButton(type: .system)
    private let collectionView: UICollectionView

    private let libraryService = PhotoLibraryService.shared
    private let analysisCoordinator = PhotoAnalysisCoordinator.shared

    private var assets: [PHAsset] = []
    private var observers: [NSObjectProtocol] = []
    private var progressContainerHeightConstraint: Constraint?
    private var delayedAnalysisStartTask: Task<Void, Never>?
    private var hasTriggeredAnalysisStart = false
    private var lastKnownStatuses: [String: PhotoAnalysisStatus] = [:]
    private var needsInitialAnalysisStatusSync = true

    init() {
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .vertical
        layout.minimumInteritemSpacing = Layout.spacing
        layout.minimumLineSpacing = Layout.spacing
        self.collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        super.init(nibName: nil, bundle: nil)
        self.title = "首页"
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        configureUI()
        registerObservers()
        reloadState()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        scheduleDelayedAnalysisStartIfNeeded()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        cancelDelayedAnalysisStart()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        updateItemSize()
    }

    private func configureUI() {
        view.backgroundColor = .systemBackground

        progressContainerView.backgroundColor = .secondarySystemBackground
        progressContainerView.layer.cornerRadius = 16
        progressContainerView.layer.masksToBounds = true
        progressContainerView.isHidden = true
        progressContainerView.alpha = 0
        progressContainerView.transform = CGAffineTransform(translationX: 0, y: -Layout.progressAnimationOffset)

        progressTitleLabel.text = "分析进度"
        progressTitleLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        progressValueLabel.font = .monospacedDigitSystemFont(ofSize: 15, weight: .semibold)
        progressValueLabel.textAlignment = .right
        progressEtaLabel.font = .systemFont(ofSize: 13, weight: .regular)
        progressEtaLabel.textColor = .secondaryLabel
        progressEtaLabel.numberOfLines = 1

        progressView.trackTintColor = .tertiarySystemFill
        progressView.progressTintColor = .systemBlue

        unauthorizedTitleLabel.text = "需要访问系统相册"
        unauthorizedTitleLabel.font = .systemFont(ofSize: 24, weight: .bold)
        unauthorizedTitleLabel.textAlignment = .center
        unauthorizedTitleLabel.numberOfLines = 0

        unauthorizedDetailLabel.text = "授权后首页会按最新时间展示所有照片，并在后台持续补齐 VL 分析结果。"
        unauthorizedDetailLabel.font = .systemFont(ofSize: 15, weight: .regular)
        unauthorizedDetailLabel.textColor = .secondaryLabel
        unauthorizedDetailLabel.textAlignment = .center
        unauthorizedDetailLabel.numberOfLines = 0

        authorizeButton.configuration = .filled()
        authorizeButton.configuration?.title = "授权访问系统相册"
        authorizeButton.addTarget(self, action: #selector(didTapAuthorizeButton), for: .touchUpInside)

        collectionView.backgroundColor = .clear
        collectionView.alwaysBounceVertical = true
        collectionView.dataSource = self
        collectionView.delegate = self
        collectionView.prefetchDataSource = self
        collectionView.contentInset = UIEdgeInsets(
            top: Layout.spacing,
            left: Layout.spacing,
            bottom: Layout.spacing,
            right: Layout.spacing
        )
        collectionView.register(PhotoGridCell.self, forCellWithReuseIdentifier: PhotoGridCell.reuseIdentifier)

        view.addSubview(scrollContainerView)
        [progressContainerView, unauthorizedContainerView, collectionView].forEach {
            scrollContainerView.addSubview($0)
        }
        [progressTitleLabel, progressValueLabel, progressEtaLabel, progressView].forEach {
            progressContainerView.addSubview($0)
        }
        [unauthorizedTitleLabel, unauthorizedDetailLabel, authorizeButton].forEach {
            unauthorizedContainerView.addSubview($0)
        }

        scrollContainerView.snp.makeConstraints { make in
            make.edges.equalTo(view.safeAreaLayoutGuide)
        }

        progressContainerView.snp.makeConstraints { make in
            make.top.equalToSuperview().inset(Layout.spacing)
            make.leading.trailing.equalToSuperview().inset(Layout.spacing)
            progressContainerHeightConstraint = make.height.equalTo(0).constraint
        }

        progressTitleLabel.snp.makeConstraints { make in
            make.top.leading.equalToSuperview().inset(14)
        }

        progressValueLabel.snp.makeConstraints { make in
            make.centerY.equalTo(progressTitleLabel)
            make.trailing.equalToSuperview().inset(14)
            make.leading.greaterThanOrEqualTo(progressTitleLabel.snp.trailing).offset(12)
        }

        progressEtaLabel.snp.makeConstraints { make in
            make.top.equalTo(progressTitleLabel.snp.bottom).offset(6)
            make.leading.equalTo(progressTitleLabel)
            make.trailing.lessThanOrEqualToSuperview().inset(14)
        }

        progressView.snp.makeConstraints { make in
            make.top.equalTo(progressEtaLabel.snp.bottom).offset(10)
            make.leading.trailing.bottom.equalToSuperview().inset(14)
        }

        unauthorizedContainerView.snp.makeConstraints { make in
            make.top.equalTo(progressContainerView.snp.bottom).offset(Layout.spacing)
            make.leading.trailing.bottom.equalToSuperview().inset(Layout.spacing)
        }

        unauthorizedTitleLabel.snp.makeConstraints { make in
            make.centerX.equalToSuperview()
            make.top.equalToSuperview().inset(80)
            make.leading.trailing.equalToSuperview().inset(28)
        }

        unauthorizedDetailLabel.snp.makeConstraints { make in
            make.top.equalTo(unauthorizedTitleLabel.snp.bottom).offset(12)
            make.leading.trailing.equalTo(unauthorizedTitleLabel)
        }

        authorizeButton.snp.makeConstraints { make in
            make.top.equalTo(unauthorizedDetailLabel.snp.bottom).offset(20)
            make.centerX.equalToSuperview()
            make.height.equalTo(52)
            make.leading.trailing.equalToSuperview().inset(40)
        }

        collectionView.snp.makeConstraints { make in
            make.top.equalTo(progressContainerView.snp.bottom).offset(Layout.spacing)
            make.leading.trailing.bottom.equalToSuperview()
        }
    }

    private func registerObservers() {
        let libraryObserver = NotificationCenter.default.addObserver(
            forName: .photoLibraryServiceDidUpdate,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.reloadState()
        }
        observers.append(libraryObserver)

        let analysisObserver = NotificationCenter.default.addObserver(
            forName: .photoAnalysisCoordinatorDidUpdate,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleAnalysisStatusUpdate()
        }
        observers.append(analysisObserver)
    }

    private func reloadState() {
        let hasAccess = libraryService.hasFullAccess
        assets = libraryService.currentAssets()

        unauthorizedContainerView.isHidden = hasAccess
        collectionView.isHidden = !hasAccess

        if hasAccess {
            updateProgress()
            collectionView.reloadData()
            scheduleDelayedAnalysisStartIfNeeded()
        } else {
            cancelDelayedAnalysisStart()
            hasTriggeredAnalysisStart = false
            lastKnownStatuses = [:]
            needsInitialAnalysisStatusSync = true
            analysisCoordinator.setProcessingEnabled(false)
            setProgressVisible(false, animated: true)
        }
    }

    private func updateProgress() {
        let snapshot = analysisCoordinator.progressSnapshot()
        guard snapshot.totalCount > 0 else {
            progressView.progress = 0
            progressValueLabel.text = nil
            progressEtaLabel.text = nil
            setProgressVisible(false, animated: true)
            return
        }

        progressValueLabel.text = "\(snapshot.processedCount) / \(snapshot.totalCount)"
        progressEtaLabel.text = snapshot.etaText ?? "正在估算剩余时长"
        progressView.progress = Float(snapshot.processedCount) / Float(snapshot.totalCount)
        setProgressVisible(snapshot.isVisible, animated: true)
    }

    private func updateItemSize() {
        guard let layout = collectionView.collectionViewLayout as? UICollectionViewFlowLayout else {
            return
        }

        let horizontalInsets = collectionView.contentInset.left + collectionView.contentInset.right
        let availableWidth = collectionView.bounds.width - horizontalInsets - Layout.spacing * 2
        let itemWidth = floor(availableWidth / 3)
        if itemWidth > 0, layout.itemSize.width != itemWidth {
            layout.itemSize = CGSize(width: itemWidth, height: itemWidth)
            layout.invalidateLayout()
            collectionView.reloadData()
        }
    }

    @objc
    private func didTapAuthorizeButton() {
        Task { [weak self] in
            guard let self else { return }
            let granted = await libraryService.requestFullAccess()
            if granted {
                libraryService.refreshAssetsIfAuthorized()
                hasTriggeredAnalysisStart = false
                scheduleDelayedAnalysisStartIfNeeded()
            }
            self.reloadState()
        }
    }
}

extension HomeViewController: UICollectionViewDataSource {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        assets.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        guard
            let cell = collectionView.dequeueReusableCell(
                withReuseIdentifier: PhotoGridCell.reuseIdentifier,
                for: indexPath
            ) as? PhotoGridCell
        else {
            return UICollectionViewCell()
        }

        let asset = assets[indexPath.item]
        cell.representedAssetIdentifier = asset.localIdentifier
        let currentStatus = analysisCoordinator.status(for: asset.localIdentifier)
        cell.apply(status: currentStatus, animateCompletion: false)
        lastKnownStatuses[asset.localIdentifier] = currentStatus

        if cell.imageRequestID != PHInvalidImageRequestID {
            libraryService.cancelImageRequest(cell.imageRequestID)
        }

        let targetSize = thumbnailTargetSize()
        cell.imageRequestID = libraryService.requestThumbnail(for: asset, targetSize: targetSize) { [weak cell] image in
            guard let cell, cell.representedAssetIdentifier == asset.localIdentifier else { return }
            cell.setThumbnail(image)
        }

        return cell
    }
}

extension HomeViewController: UICollectionViewDelegateFlowLayout {
    func collectionView(_ collectionView: UICollectionView, didEndDisplaying cell: UICollectionViewCell, forItemAt indexPath: IndexPath) {
        guard let photoCell = cell as? PhotoGridCell else { return }
        if photoCell.imageRequestID != PHInvalidImageRequestID {
            libraryService.cancelImageRequest(photoCell.imageRequestID)
            photoCell.imageRequestID = PHInvalidImageRequestID
        }
    }
}

extension HomeViewController: UICollectionViewDataSourcePrefetching {
    func collectionView(_ collectionView: UICollectionView, prefetchItemsAt indexPaths: [IndexPath]) {
        let assetsToCache = indexPaths.compactMap { indexPath in
            assets.indices.contains(indexPath.item) ? assets[indexPath.item] : nil
        }
        guard !assetsToCache.isEmpty else { return }
        libraryService.startCaching(for: assetsToCache, targetSize: thumbnailTargetSize())
    }

    func collectionView(_ collectionView: UICollectionView, cancelPrefetchingForItemsAt indexPaths: [IndexPath]) {
        let assetsToStopCaching = indexPaths.compactMap { indexPath in
            assets.indices.contains(indexPath.item) ? assets[indexPath.item] : nil
        }
        guard !assetsToStopCaching.isEmpty else { return }
        libraryService.stopCaching(for: assetsToStopCaching, targetSize: thumbnailTargetSize())
    }
}

private extension HomeViewController {
    func handleAnalysisStatusUpdate() {
        updateProgress()
        if needsInitialAnalysisStatusSync {
            applyVisibleCellStatusesSilently()
            lastKnownStatuses = captureCurrentStatuses()
            needsInitialAnalysisStatusSync = false
            return
        }
        refreshVisibleCellStatuses()
        lastKnownStatuses = captureCurrentStatuses()
    }

    func scheduleDelayedAnalysisStartIfNeeded() {
        guard isViewLoaded, view.window != nil else { return }
        guard libraryService.hasFullAccess else { return }
        guard !hasTriggeredAnalysisStart else { return }
        guard delayedAnalysisStartTask == nil else { return }

        analysisCoordinator.start()
        delayedAnalysisStartTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: Layout.analysisStartDelayNanoseconds)
            } catch {
                return
            }

            await MainActor.run {
                guard let self else { return }
                guard self.isViewLoaded, self.view.window != nil else {
                    self.delayedAnalysisStartTask = nil
                    return
                }
                guard self.libraryService.hasFullAccess else {
                    self.delayedAnalysisStartTask = nil
                    return
                }

                self.hasTriggeredAnalysisStart = true
                self.delayedAnalysisStartTask = nil
                self.analysisCoordinator.setProcessingEnabled(true)
                self.updateProgress()
            }
        }
    }

    func cancelDelayedAnalysisStart() {
        delayedAnalysisStartTask?.cancel()
        delayedAnalysisStartTask = nil
    }

    func setProgressVisible(_ visible: Bool, animated: Bool) {
        let alreadyVisible = !progressContainerView.isHidden && progressContainerView.alpha > 0.99
        let alreadyHidden = progressContainerView.isHidden || progressContainerView.alpha < 0.01

        if visible, alreadyVisible {
            progressContainerHeightConstraint?.update(offset: Layout.progressExpandedHeight)
            progressContainerView.transform = .identity
            progressContainerView.alpha = 1
            return
        }

        if !visible, alreadyHidden {
            progressContainerView.isHidden = true
            progressContainerHeightConstraint?.update(offset: 0)
            progressContainerView.alpha = 0
            progressContainerView.transform = CGAffineTransform(translationX: 0, y: -Layout.progressAnimationOffset)
            return
        }

        let animations = {
            self.progressContainerHeightConstraint?.update(offset: visible ? Layout.progressExpandedHeight : 0)
            self.progressContainerView.alpha = visible ? 1 : 0
            self.progressContainerView.transform = visible
                ? .identity
                : CGAffineTransform(translationX: 0, y: -Layout.progressAnimationOffset)
            self.view.layoutIfNeeded()
        }

        if visible {
            progressContainerView.isHidden = false
        }

        guard animated else {
            animations()
            if !visible {
                progressContainerView.isHidden = true
            }
            return
        }

        UIView.animate(
            withDuration: Layout.progressAnimationDuration,
            delay: 0,
            options: [.curveEaseInOut, .beginFromCurrentState]
        ) {
            animations()
        } completion: { _ in
            if !visible {
                self.progressContainerView.isHidden = true
            }
        }
    }

    func refreshVisibleCellStatuses() {
        for cell in collectionView.visibleCells {
            guard let photoCell = cell as? PhotoGridCell else { continue }
            guard let assetIdentifier = photoCell.representedAssetIdentifier else { continue }
            let currentStatus = analysisCoordinator.status(for: assetIdentifier)
            let previousStatus = lastKnownStatuses[assetIdentifier]
            if currentStatus == .completed {
                if previousStatus != .completed {
                    photoCell.apply(status: .completed, animateCompletion: true)
                }
                continue
            }
            photoCell.apply(status: currentStatus, animateCompletion: false)
        }
    }

    func applyVisibleCellStatusesSilently() {
        for cell in collectionView.visibleCells {
            guard let photoCell = cell as? PhotoGridCell else { continue }
            guard let assetIdentifier = photoCell.representedAssetIdentifier else { continue }
            let currentStatus = analysisCoordinator.status(for: assetIdentifier)
            photoCell.apply(status: currentStatus, animateCompletion: false)
        }
    }

    func captureCurrentStatuses() -> [String: PhotoAnalysisStatus] {
        Dictionary(uniqueKeysWithValues: assets.map { asset in
            (asset.localIdentifier, analysisCoordinator.status(for: asset.localIdentifier))
        })
    }

    func thumbnailTargetSize() -> CGSize {
        guard let layout = collectionView.collectionViewLayout as? UICollectionViewFlowLayout else {
            return CGSize(width: 200, height: 200)
        }
        let scale = UIScreen.main.scale
        return CGSize(width: layout.itemSize.width * scale, height: layout.itemSize.height * scale)
    }
}
