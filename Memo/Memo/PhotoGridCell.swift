import Photos
import SnapKit
import UIKit

final class PhotoGridCell: UICollectionViewCell {
    static let reuseIdentifier = "PhotoGridCell"

    private let imageView = UIImageView()
    private let statusContainerView = UIView()
    private let statusLabel = UILabel()

    var representedAssetIdentifier: String?
    var imageRequestID: PHImageRequestID = PHInvalidImageRequestID

    override init(frame: CGRect) {
        super.init(frame: frame)
        configureUI()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func prepareForReuse() {
        super.prepareForReuse()
        representedAssetIdentifier = nil
        imageRequestID = PHInvalidImageRequestID
        imageView.image = nil
        apply(status: .pending)
    }

    func setThumbnail(_ image: UIImage?) {
        imageView.image = image
    }

    func apply(status: PhotoAnalysisStatus) {
        switch status {
        case .pending:
            statusLabel.text = "待分析"
            statusContainerView.backgroundColor = UIColor.secondaryLabel.withAlphaComponent(0.72)
        case .running:
            statusLabel.text = "分析中"
            statusContainerView.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.84)
        case .completed:
            statusLabel.text = "已完成"
            statusContainerView.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.84)
        case .failed:
            statusLabel.text = "失败"
            statusContainerView.backgroundColor = UIColor.systemRed.withAlphaComponent(0.84)
        }
    }

    private func configureUI() {
        contentView.backgroundColor = .secondarySystemBackground
        contentView.layer.cornerRadius = 14
        contentView.layer.masksToBounds = true

        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true

        statusContainerView.layer.cornerRadius = 9
        statusContainerView.layer.masksToBounds = true

        statusLabel.font = .systemFont(ofSize: 11, weight: .semibold)
        statusLabel.textColor = .white
        statusLabel.textAlignment = .center

        contentView.addSubview(imageView)
        contentView.addSubview(statusContainerView)
        statusContainerView.addSubview(statusLabel)

        imageView.snp.makeConstraints { make in
            make.edges.equalToSuperview()
        }

        statusContainerView.snp.makeConstraints { make in
            make.leading.bottom.equalToSuperview().inset(8)
        }

        statusLabel.snp.makeConstraints { make in
            make.edges.equalToSuperview().inset(UIEdgeInsets(top: 4, left: 8, bottom: 4, right: 8))
        }
    }
}
