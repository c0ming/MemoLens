import Photos
import SnapKit
import UIKit

final class PhotoGridCell: UICollectionViewCell {
    static let reuseIdentifier = "PhotoGridCell"

    private enum Layout {
        static let badgeInset: CGFloat = 8
        static let badgeHeight: CGFloat = 24
        static let badgeHorizontalPadding: CGFloat = 8
        static let badgeCircleSize: CGFloat = 24
        static let badgeAnimationDuration: TimeInterval = 0.24
        static let badgeFadeDelay: TimeInterval = 0.38
        static let badgeFadeDuration: TimeInterval = 0.5
    }

    private let imageView = UIImageView()
    private let statusContainerView = UIView()
    private let statusLabel = UILabel()
    private var statusWidthConstraint: Constraint?
    private var statusHeightConstraint: Constraint?

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
        statusContainerView.layer.removeAllAnimations()
        statusContainerView.alpha = 1
        statusContainerView.isHidden = false
        statusContainerView.transform = .identity
        apply(status: .pending, animateCompletion: false)
    }

    func apply(status: PhotoAnalysisStatus, animateCompletion: Bool) {
        if status != .completed {
            statusContainerView.layer.removeAllAnimations()
            statusContainerView.alpha = 1
            statusContainerView.isHidden = false
            statusContainerView.transform = .identity
        }

        switch status {
        case .pending:
            applyPillStyle(
                text: "待分析",
                backgroundColor: UIColor.secondaryLabel.withAlphaComponent(0.72)
            )
        case .running:
            applyPillStyle(
                text: "分析中",
                backgroundColor: UIColor.systemBlue.withAlphaComponent(0.84)
            )
        case .completed:
            if animateCompletion {
                animateCompletionBadgeIfNeeded()
            } else {
                applyCompletedStyleWithoutAnimation()
            }
        case .failed:
            applyPillStyle(
                text: "失败",
                backgroundColor: UIColor.systemRed.withAlphaComponent(0.84)
            )
        }
    }

    func setThumbnail(_ image: UIImage?) {
        imageView.image = image
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
            make.top.trailing.equalToSuperview().inset(Layout.badgeInset)
            statusWidthConstraint = make.width.equalTo(0).constraint
            statusHeightConstraint = make.height.equalTo(Layout.badgeHeight).constraint
        }

        statusLabel.snp.makeConstraints { make in
            make.center.equalToSuperview()
        }
    }

    private func applyPillStyle(text: String, backgroundColor: UIColor) {
        statusLabel.text = text
        statusLabel.font = .systemFont(ofSize: 11, weight: .semibold)
        statusContainerView.backgroundColor = backgroundColor
        statusContainerView.layer.cornerRadius = Layout.badgeHeight / 2
        statusWidthConstraint?.update(offset: badgeWidth(for: text))
        statusHeightConstraint?.update(offset: Layout.badgeHeight)
    }

    private func applyCompletedStyleWithoutAnimation() {
        statusContainerView.layer.removeAllAnimations()
        statusContainerView.isHidden = true
        statusContainerView.alpha = 0
        statusContainerView.transform = .identity
        statusContainerView.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.96)
        statusContainerView.layer.cornerRadius = Layout.badgeCircleSize / 2
        statusWidthConstraint?.update(offset: Layout.badgeCircleSize)
        statusHeightConstraint?.update(offset: Layout.badgeCircleSize)
        statusLabel.font = .systemFont(ofSize: 12, weight: .bold)
        statusLabel.text = "✓"
    }

    private func animateCompletionBadgeIfNeeded() {
        statusContainerView.layer.removeAllAnimations()
        statusContainerView.alpha = 1
        statusContainerView.isHidden = false
        statusContainerView.transform = .identity
        statusContainerView.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.9)

        UIView.animate(
            withDuration: Layout.badgeAnimationDuration,
            delay: 0,
            options: [.curveEaseInOut, .beginFromCurrentState]
        ) { [weak self] in
            guard let self else { return }
            self.statusWidthConstraint?.update(offset: Layout.badgeCircleSize)
            self.statusHeightConstraint?.update(offset: Layout.badgeCircleSize)
            self.statusContainerView.layer.cornerRadius = Layout.badgeCircleSize / 2
            self.statusContainerView.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.96)
            self.layoutIfNeeded()
        } completion: { [weak self] _ in
            guard let self else { return }
            UIView.transition(
                with: self.statusLabel,
                duration: 0.16,
                options: [.transitionCrossDissolve, .beginFromCurrentState],
                animations: {
                self.statusLabel.font = .systemFont(ofSize: 12, weight: .bold)
                self.statusLabel.text = "✓"
            },
                completion: nil
            )

            UIView.animate(
                withDuration: Layout.badgeFadeDuration,
                delay: Layout.badgeFadeDelay,
                options: [.curveEaseInOut, .beginFromCurrentState]
            ) { [weak self] in
                self?.statusContainerView.alpha = 0
                self?.statusContainerView.transform = CGAffineTransform(scaleX: 0.92, y: 0.92)
            } completion: { [weak self] _ in
                self?.statusContainerView.isHidden = true
            }
        }
    }

    private func badgeWidth(for text: String) -> CGFloat {
        let textWidth = (text as NSString).size(withAttributes: [
            .font: UIFont.systemFont(ofSize: 11, weight: .semibold),
        ]).width
        return ceil(textWidth) + Layout.badgeHorizontalPadding * 2
    }
}
