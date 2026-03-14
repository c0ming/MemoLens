import Photos
import SnapKit
import UIKit

final class HomeViewController: UIViewController {
    private enum Layout {
        static let spacing: CGFloat = 12
    }

    private let scrollContainerView = UIView()
    private let progressContainerView = UIView()
    private let progressTitleLabel = UILabel()
    private let progressValueLabel = UILabel()
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

        progressTitleLabel.text = "分析进度"
        progressTitleLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        progressValueLabel.font = .monospacedDigitSystemFont(ofSize: 15, weight: .semibold)
        progressValueLabel.textAlignment = .right

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
        [progressTitleLabel, progressValueLabel, progressView].forEach {
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
        }

        progressTitleLabel.snp.makeConstraints { make in
            make.top.leading.equalToSuperview().inset(14)
        }

        progressValueLabel.snp.makeConstraints { make in
            make.centerY.equalTo(progressTitleLabel)
            make.trailing.equalToSuperview().inset(14)
            make.leading.greaterThanOrEqualTo(progressTitleLabel.snp.trailing).offset(12)
        }

        progressView.snp.makeConstraints { make in
            make.top.equalTo(progressTitleLabel.snp.bottom).offset(12)
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
            self?.updateProgress()
            self?.collectionView.reloadData()
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
            analysisCoordinator.start()
        } else {
            progressContainerView.isHidden = true
        }
    }

    private func updateProgress() {
        let snapshot = analysisCoordinator.progressSnapshot()
        progressContainerView.isHidden = !snapshot.isVisible
        guard snapshot.totalCount > 0 else {
            progressView.progress = 0
            progressValueLabel.text = nil
            return
        }

        progressValueLabel.text = "\(snapshot.processedCount) / \(snapshot.totalCount)"
        progressView.progress = Float(snapshot.processedCount) / Float(snapshot.totalCount)
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
                analysisCoordinator.start()
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
        cell.apply(status: analysisCoordinator.status(for: asset.localIdentifier))

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
    func thumbnailTargetSize() -> CGSize {
        guard let layout = collectionView.collectionViewLayout as? UICollectionViewFlowLayout else {
            return CGSize(width: 200, height: 200)
        }
        let scale = UIScreen.main.scale
        return CGSize(width: layout.itemSize.width * scale, height: layout.itemSize.height * scale)
    }
}
