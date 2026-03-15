//
//  AppDelegate.swift
//  Memo
//
//  Created by c0mingxx on 2026/3/14.
//

import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        #if targetEnvironment(simulator)
        Task { @MainActor in
            await FeaturedMemoryWidgetSync.shared.refreshSimulatorPreview()
        }
        #endif
        return true
    }

    func applicationDidBecomeActive(_ application: UIApplication) {
        PhotoAnalysisCoordinator.shared.sceneDidBecomeActive()
        #if !targetEnvironment(simulator)
        Task {
            await LocalVLMService.shared.applicationDidBecomeActive()
        }
        #endif
    }

    func applicationWillResignActive(_ application: UIApplication) {
        PhotoAnalysisCoordinator.shared.sceneWillResignActive()
        synchronouslyFreezeVLMForBackground()
    }

    func applicationDidEnterBackground(_ application: UIApplication) {
        PhotoAnalysisCoordinator.shared.sceneDidEnterBackground()
        synchronouslyFreezeVLMForBackground()
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
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
