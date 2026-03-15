//
//  SceneDelegate.swift
//  Memo
//
//  Created by c0mingxx on 2026/3/14.
//

import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = RootTabBarController()
        window.makeKeyAndVisible()
        self.window = window

        PhotoLibraryService.shared.refreshAssetsIfAuthorized()
        PhotoAnalysisCoordinator.shared.start()
    }

    func sceneDidDisconnect(_ scene: UIScene) {
    }

    func sceneDidBecomeActive(_ scene: UIScene) {
        PhotoAnalysisCoordinator.shared.sceneDidBecomeActive()
        #if !targetEnvironment(simulator)
        Task {
            await LocalVLMService.shared.applicationDidBecomeActive()
        }
        #endif
    }

    func sceneWillResignActive(_ scene: UIScene) {
        PhotoAnalysisCoordinator.shared.sceneWillResignActive()
        synchronouslyFreezeVLMForBackground()
    }

    func sceneWillEnterForeground(_ scene: UIScene) {
        PhotoLibraryService.shared.refreshAssetsIfAuthorized()
    }

    func sceneDidEnterBackground(_ scene: UIScene) {
        PhotoAnalysisCoordinator.shared.sceneDidEnterBackground()
        synchronouslyFreezeVLMForBackground()
    }

    private func synchronouslyFreezeVLMForBackground() {
        #if targetEnvironment(simulator)
        return
        #else
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            await LocalVLMService.shared.applicationWillResignActive()
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 0.35)
        #endif
    }
}
