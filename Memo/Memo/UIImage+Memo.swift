import UIKit

extension UIImage {
    func resizedToFit(maxPixelArea: CGFloat, maxLongEdge: CGFloat? = nil) -> UIImage {
        let originalWidth = size.width
        let originalHeight = size.height
        guard originalWidth > 0, originalHeight > 0 else { return self }

        var scaleFactor: CGFloat = 1
        let pixelArea = originalWidth * originalHeight
        if pixelArea > maxPixelArea {
            scaleFactor = min(scaleFactor, sqrt(maxPixelArea / pixelArea))
        }

        if let maxLongEdge {
            let longEdge = max(originalWidth, originalHeight)
            if longEdge > maxLongEdge {
                scaleFactor = min(scaleFactor, maxLongEdge / longEdge)
            }
        }

        guard scaleFactor < 0.999 else { return self }

        let targetSize = CGSize(
            width: floor(originalWidth * scaleFactor),
            height: floor(originalHeight * scaleFactor)
        )

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resizedImage = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: targetSize))
        }
        print(
            "[UIImage+Memo] resizedToFit original=\(Int(originalWidth))x\(Int(originalHeight)) "
                + "target=\(Int(targetSize.width))x\(Int(targetSize.height))"
        )
        return resizedImage
    }

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
