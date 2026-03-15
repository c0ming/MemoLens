import SwiftUI
import UIKit
import WidgetKit

private struct FeaturedMemoryEntry: TimelineEntry {
    let date: Date
    let payload: MemoryWidgetPayload
    let image: UIImage?
}

private struct FeaturedMemoryProvider: TimelineProvider {
    private static let simulatorCaption = "湖面安静下来时，午后的光也慢慢落进了回忆里。"

    func placeholder(in context: Context) -> FeaturedMemoryEntry {
        FeaturedMemoryEntry(
            date: Date(),
            payload: MemoryWidgetPayload(
                state: .ready,
                assetLocalIdentifier: nil,
                captionLine: "风吹过时，回忆也有了形状。",
                imageFileName: nil,
                updatedAt: Date().timeIntervalSince1970
            ),
            image: nil
        )
    }

    func getSnapshot(in context: Context, completion: @escaping (FeaturedMemoryEntry) -> Void) {
        completion(loadEntry(for: context.family))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<FeaturedMemoryEntry>) -> Void) {
        let entry = loadEntry(for: context.family)
        let refreshDate = Calendar.current.date(byAdding: .minute, value: 30, to: Date()) ?? Date().addingTimeInterval(1800)
        completion(Timeline(entries: [entry], policy: .after(refreshDate)))
    }

    private func loadEntry(for family: WidgetFamily) -> FeaturedMemoryEntry {
        #if targetEnvironment(simulator)
        return loadSimulatorEntry(for: family)
        #else
        let payload = loadPayload()
        let image = loadImage(fileName: payload.imageFileName, family: family)
        return FeaturedMemoryEntry(date: Date(), payload: payload, image: image)
        #endif
    }

    #if targetEnvironment(simulator)
    private func loadSimulatorEntry(for family: WidgetFamily) -> FeaturedMemoryEntry {
        let payload = MemoryWidgetPayload(
            state: .ready,
            assetLocalIdentifier: "simulator-preview",
            captionLine: Self.simulatorCaption,
            imageFileName: MemoryWidgetShared.imageFileName,
            updatedAt: Date().timeIntervalSince1970
        )
        let image = loadImage(fileName: MemoryWidgetShared.imageFileName, family: family) ?? makeFallbackMockImage(for: family)
        return FeaturedMemoryEntry(date: Date(), payload: payload, image: image)
    }
    #endif

    private func loadPayload() -> MemoryWidgetPayload {
        guard
            let containerURL = MemoryWidgetShared.sharedContainerURL(),
            let data = try? Data(contentsOf: containerURL.appendingPathComponent(MemoryWidgetShared.payloadFileName)),
            let payload = try? JSONDecoder().decode(MemoryWidgetPayload.self, from: data)
        else {
            return MemoryWidgetPayload(
                state: .empty,
                assetLocalIdentifier: nil,
                captionLine: "还没有可展示的照片",
                imageFileName: nil,
                updatedAt: Date().timeIntervalSince1970
            )
        }
        return payload
    }

    private func loadImage(fileName: String?, family: WidgetFamily) -> UIImage? {
        guard
            let fileName,
            let containerURL = MemoryWidgetShared.sharedContainerURL(),
            let data = try? Data(contentsOf: containerURL.appendingPathComponent(fileName))
        else {
            return nil
        }
        guard let image = UIImage(data: data) else {
            return nil
        }
        return prepareImage(image, family: family)
    }

    private func prepareImage(_ image: UIImage, family: WidgetFamily) -> UIImage {
        let originalWidth = image.size.width
        let originalHeight = image.size.height
        guard originalWidth > 0, originalHeight > 0 else { return image }

        let limits = widgetImageLimits(for: family)

        var scaleFactor: CGFloat = 1
        let pixelArea = originalWidth * originalHeight
        if pixelArea > limits.maxPixelArea {
            scaleFactor = min(scaleFactor, sqrt(limits.maxPixelArea / pixelArea))
        }

        let longEdge = max(originalWidth, originalHeight)
        if longEdge > limits.maxLongEdge {
            scaleFactor = min(scaleFactor, limits.maxLongEdge / longEdge)
        }

        guard scaleFactor < 0.999 else { return image }

        let targetSize = CGSize(
            width: floor(originalWidth * scaleFactor),
            height: floor(originalHeight * scaleFactor)
        )
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    private func makeFallbackMockImage(for family: WidgetFamily) -> UIImage? {
        let limits = widgetImageLimits(for: family)
        let targetSize = CGSize(width: limits.mockWidth, height: limits.mockHeight)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { context in
            let rect = CGRect(origin: .zero, size: targetSize)
            let cgContext = context.cgContext

            let colors = [
                UIColor(red: 0.85, green: 0.92, blue: 0.98, alpha: 1).cgColor,
                UIColor(red: 0.73, green: 0.83, blue: 0.73, alpha: 1).cgColor,
            ] as CFArray
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let gradient = CGGradient(colorsSpace: colorSpace, colors: colors, locations: [0, 1])
            cgContext.drawLinearGradient(
                gradient!,
                start: CGPoint(x: 0, y: 0),
                end: CGPoint(x: targetSize.width, y: targetSize.height),
                options: []
            )

            UIColor.white.withAlphaComponent(0.85).setFill()
            cgContext.fillEllipse(in: CGRect(x: 120, y: 130, width: 660, height: 660))

            UIColor(red: 0.40, green: 0.31, blue: 0.20, alpha: 1).setFill()
            cgContext.fillEllipse(in: CGRect(x: 170, y: 660, width: 560, height: 300))
        }
    }

    private func widgetImageLimits(for family: WidgetFamily) -> (
        maxPixelArea: CGFloat,
        maxLongEdge: CGFloat,
        mockWidth: CGFloat,
        mockHeight: CGFloat
    ) {
        switch family {
        case .systemSmall:
            return (900_000, 1_050, 720, 960)
        case .systemMedium:
            return (1_250_000, 1_250, 840, 1_080)
        case .systemLarge:
            return (1_600_000, 1_400, 900, 1_200)
        default:
            return (1_100_000, 1_150, 800, 1_040)
        }
    }
}

private struct FeaturedMemoryWidgetView: View {
    @Environment(\.widgetFamily) private var family

    let entry: FeaturedMemoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            imageSection
            captionSection
                .padding(.top, 12)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .containerBackground(for: .widget) {
            Color(uiColor: .secondarySystemBackground)
        }
    }

    @ViewBuilder
    private var imageSection: some View {
        let shape = RoundedRectangle(cornerRadius: 16, style: .continuous)
        Color.clear
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay {
                if let image = entry.image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Color(uiColor: .tertiarySystemFill)
                        .overlay {
                            Image(systemName: "photo")
                                .font(.system(size: iconSize, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                }
            }
            .clipShape(shape)
    }

    private var captionSection: some View {
        Text(entry.payload.captionLine)
            .font(captionFont)
            .foregroundStyle(.primary)
            .lineLimit(2)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var iconSize: CGFloat {
        switch family {
        case .systemSmall:
            return 20
        case .systemMedium:
            return 22
        case .systemLarge:
            return 24
        default:
            return 22
        }
    }

    private var captionFont: Font {
        switch family {
        case .systemSmall:
            return .system(size: 11, weight: .medium, design: .rounded)
        case .systemMedium:
            return .system(size: 13, weight: .medium, design: .rounded)
        case .systemLarge:
            return .system(size: 15, weight: .medium, design: .rounded)
        default:
            return .system(size: 13, weight: .medium, design: .rounded)
        }
    }


}

struct MemoFeaturedMemoryWidget: Widget {
    let kind: String = MemoryWidgetShared.widgetKind

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: FeaturedMemoryProvider()) { entry in
            FeaturedMemoryWidgetView(entry: entry)
        }
        .contentMarginsDisabled()
        .configurationDisplayName("今日回忆")
        .description("显示一张高回忆度照片和题注。")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
}

@main
struct MemoWidgetBundle: WidgetBundle {
    var body: some Widget {
        MemoFeaturedMemoryWidget()
    }
}
