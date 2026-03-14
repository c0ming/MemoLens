import UIKit

final class RootTabBarController: UITabBarController {
    override func viewDidLoad() {
        super.viewDidLoad()
        configureTabs()
    }

    private func configureTabs() {
        let homeNavigationController = makeNavigationController(
            rootViewController: HomeViewController(),
            title: "首页",
            imageName: "house",
            selectedImageName: "house.fill"
        )
        let testNavigationController = makeNavigationController(
            rootViewController: LocalVLMTestViewController(),
            title: "测试",
            imageName: "sparkles.rectangle.stack",
            selectedImageName: "sparkles.rectangle.stack.fill"
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
            homeNavigationController,
            testNavigationController,
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
