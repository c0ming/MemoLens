import Foundation

enum MemoryWidgetContentState: String, Codable {
    case ready
    case empty
}

struct MemoryWidgetPayload: Codable {
    let state: MemoryWidgetContentState
    let assetLocalIdentifier: String?
    let captionLine: String
    let imageFileName: String?
    let updatedAt: TimeInterval
}

enum MemoryWidgetShared {
    static let widgetKind = "MemoFeaturedMemoryWidget"
    static let appGroupIdentifier = "group.ai.octopus.Memo"
    static let payloadFileName = "featured_memory_widget.json"
    static let imageFileName = "featured_memory_widget.jpg"

    static func sharedContainerURL() -> URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupIdentifier)
    }
}
