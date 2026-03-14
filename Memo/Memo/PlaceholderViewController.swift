import SnapKit
import UIKit

final class PlaceholderViewController: UIViewController {
    private let messageLabel = UILabel()
    private let detailLabel = UILabel()

    private let headline: String
    private let detail: String

    init(title: String, headline: String, detail: String) {
        self.headline = headline
        self.detail = detail
        super.init(nibName: nil, bundle: nil)
        self.title = title
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        configureUI()
    }

    private func configureUI() {
        view.backgroundColor = .systemBackground

        messageLabel.text = headline
        messageLabel.font = .systemFont(ofSize: 28, weight: .bold)
        messageLabel.numberOfLines = 0
        messageLabel.textAlignment = .center

        detailLabel.text = detail
        detailLabel.font = .systemFont(ofSize: 15, weight: .regular)
        detailLabel.textColor = .secondaryLabel
        detailLabel.numberOfLines = 0
        detailLabel.textAlignment = .center

        [messageLabel, detailLabel].forEach {
            view.addSubview($0)
        }

        messageLabel.snp.makeConstraints { make in
            make.centerX.equalToSuperview()
            make.centerY.equalToSuperview().offset(-18)
            make.leading.trailing.equalTo(view.safeAreaLayoutGuide).inset(28)
        }

        detailLabel.snp.makeConstraints { make in
            make.top.equalTo(messageLabel.snp.bottom).offset(12)
            make.leading.trailing.equalTo(messageLabel)
        }
    }
}
