import UIKit

extension UIImage {
    func normalizedJPEGData() -> Data {
        if let data = jpegData(compressionQuality: 0.92) {
            print("[UIImage+Memo] normalizedJPEGData via jpegData, bytes=\(data.count)")
            return data
        }

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let renderedImage = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
        let fallbackData = renderedImage.jpegData(compressionQuality: 0.92) ?? Data()
        print("[UIImage+Memo] normalizedJPEGData via renderer fallback, bytes=\(fallbackData.count)")
        return fallbackData
    }
}
