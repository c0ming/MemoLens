import UIKit

final class RootTabBarController: UITabBarController {
    override func viewDidLoad() {
        super.viewDidLoad()
        configureTabs()
    }

    private func configureTabs() {
        let testNavigationController = makeNavigationController(
            rootViewController: LocalVLMTestViewController(),
            title: "测试",
            imageName: "sparkles.rectangle.stack",
            selectedImageName: "sparkles.rectangle.stack.fill"
        )
        let albumsNavigationController = makeNavigationController(
            rootViewController: PlaceholderViewController(
                title: "相册",
                headline: "相册页",
                detail: "这里后续放系统相册读取、分组浏览和记忆卡片入口。"
            ),
            title: "相册",
            imageName: "photo.on.rectangle",
            selectedImageName: "photo.on.rectangle.fill"
        )
        let settingsNavigationController = makeNavigationController(
            rootViewController: PlaceholderViewController(
                title: "设置",
                headline: "设置页",
                detail: "这里后续放模型配置、设备能力检查和缓存管理。"
            ),
            title: "设置",
            imageName: "gearshape",
            selectedImageName: "gearshape.fill"
        )

        viewControllers = [
            testNavigationController,
            albumsNavigationController,
            settingsNavigationController,
        ]
    }

    private func makeNavigationController(
        rootViewController: UIViewController,
        title: String,
        imageName: String,
        selectedImageName: String
    ) -> UINavigationController {
        let navigationController = UINavigationController(rootViewController: rootViewController)
        navigationController.tabBarItem = UITabBarItem(
            title: title,
            image: UIImage(systemName: imageName),
            selectedImage: UIImage(systemName: selectedImageName)
        )
        navigationController.navigationBar.prefersLargeTitles = false
        return navigationController
    }
}
