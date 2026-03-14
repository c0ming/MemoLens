import Foundation

enum PhotoAnalysisStatus: String, Codable {
    case pending
    case running
    case completed
    case failed
}

struct PhotoAnalysisRecord: Codable {
    let assetLocalIdentifier: String
    let assetCreationTimestamp: TimeInterval?
    let assetModificationTimestamp: TimeInterval?
    let status: PhotoAnalysisStatus
    let updatedAt: TimeInterval
    let analysisDurationSeconds: TimeInterval?
    let memoryScore: Double?
    let resultJSONString: String?
    let errorMessage: String?
}

actor PhotoAnalysisStore {
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    private let decoder = JSONDecoder()
    private let directoryName = "photo_analysis"

    func loadAllRecords() async -> [String: PhotoAnalysisRecord] {
        do {
            let directoryURL = try recordsDirectoryURL()
            let fileURLs = try FileManager.default.contentsOfDirectory(
                at: directoryURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )

            var records: [String: PhotoAnalysisRecord] = [:]
            for fileURL in fileURLs where fileURL.pathExtension == "json" {
                do {
                    let data = try Data(contentsOf: fileURL)
                    let record = try decoder.decode(PhotoAnalysisRecord.self, from: data)
                    records[record.assetLocalIdentifier] = record
                } catch {
                    print("[PhotoAnalysisStore] failed to decode \(fileURL.lastPathComponent): \(error.localizedDescription)")
                }
            }
            return records
        } catch {
            print("[PhotoAnalysisStore] loadAllRecords failed: \(error.localizedDescription)")
            return [:]
        }
    }

    func save(_ record: PhotoAnalysisRecord) async {
        do {
            let fileURL = try recordFileURL(for: record.assetLocalIdentifier)
            let data = try encoder.encode(record)
            try data.write(to: fileURL, options: [.atomic])
            print("[PhotoAnalysisStore] saved record for \(record.assetLocalIdentifier)")
        } catch {
            print("[PhotoAnalysisStore] save failed for \(record.assetLocalIdentifier): \(error.localizedDescription)")
        }
    }

    private func recordsDirectoryURL() throws -> URL {
        let baseURL = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let appURL = baseURL.appendingPathComponent("Memo", isDirectory: true)
        let directoryURL = appURL.appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        return directoryURL
    }

    private func recordFileURL(for assetLocalIdentifier: String) throws -> URL {
        let fileName = sanitizedFileName(for: assetLocalIdentifier)
        return try recordsDirectoryURL().appendingPathComponent(fileName).appendingPathExtension("json")
    }

    private func sanitizedFileName(for assetLocalIdentifier: String) -> String {
        assetLocalIdentifier
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
            .replacingOccurrences(of: " ", with: "_")
    }
}
