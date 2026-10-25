import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func googleResponseJSONSchema(_ schema: JSONValue) -> JSONValue {
    guard var object = schema.objectValue else { return schema }

    if let constant = object.removeValue(forKey: "const") {
        object["enum"] = .array([constant])
    }
    if let properties = object["properties"]?.objectValue {
        object["properties"] = .object(properties.mapValues(googleResponseJSONSchema))
    }
    if let items = object["items"] {
        if let itemSchemas = items.arrayValue {
            object["items"] = .array(itemSchemas.map(googleResponseJSONSchema))
        } else {
            object["items"] = googleResponseJSONSchema(items)
        }
    }
    if let additionalProperties = object["additionalProperties"], additionalProperties.boolValue == nil {
        object["additionalProperties"] = googleResponseJSONSchema(additionalProperties)
    }
    for key in ["anyOf", "oneOf"] {
        if let alternatives = object[key]?.arrayValue {
            object[key] = .array(alternatives.map(googleResponseJSONSchema))
        }
    }
    if let definitions = object["$defs"]?.objectValue {
        object["$defs"] = .object(definitions.mapValues(googleResponseJSONSchema))
    }
    return .object(object)
}

func googleOpenAPISchema(from schema: JSONValue, isRoot: Bool) throws -> JSONValue? {
    let rootObject = schema.objectValue
    return try googleOpenAPISchema(
        from: schema,
        isRoot: isRoot,
        referenceContext: GoogleSchemaReferenceContext(
            definitions: rootObject?["definitions"]?.objectValue,
            dollarDefinitions: rootObject?["$defs"]?.objectValue,
            resolvingReferences: []
        )
    )
}

private struct GoogleSchemaReferenceContext {
    var definitions: [String: JSONValue]?
    var dollarDefinitions: [String: JSONValue]?
    var resolvingReferences: Set<String>
}

private func googleOpenAPISchema(
    from schema: JSONValue,
    isRoot: Bool,
    referenceContext: GoogleSchemaReferenceContext
) throws -> JSONValue? {
    if case .bool = schema {
        return .object(["type": .string("boolean"), "properties": .object([:])])
    }
    guard let object = schema.objectValue else { return schema }

    if let reference = object["$ref"]?.stringValue {
        return try googleOpenAPISchemaReference(
            object,
            reference: reference,
            isRoot: isRoot,
            referenceContext: referenceContext
        )
    }

    if isRoot,
       object["type"]?.stringValue == "object",
       (object["properties"]?.objectValue?.isEmpty ?? true),
       object["additionalProperties"]?.boolValue != true {
        return nil
    }

    var result: [String: JSONValue] = [:]
    for key in ["description", "required", "format", "minLength", "minItems", "maxItems"] {
        if let value = object[key] {
            result[key] = value
        }
    }

    if let type = object["type"] {
        if let types = type.arrayValue?.compactMap(\.stringValue) {
            let nonNullTypes = types.filter { $0 != "null" }
            if nonNullTypes.isEmpty {
                result["type"] = .string("null")
            } else {
                result["anyOf"] = .array(nonNullTypes.map { .object(["type": .string($0)]) })
                if types.contains("null") {
                    result["nullable"] = true
                }
            }
        } else {
            result["type"] = type
        }
    }

    let enumValues = object["enum"]?.arrayValue ?? object["const"].map { [$0] }
    if let enumValues {
        try googleAddEnumToSchema(values: enumValues, declaredType: object["type"], result: &result)
    }

    if let properties = object["properties"]?.objectValue {
        var converted: [String: JSONValue] = [:]
        for (name, propertySchema) in properties {
            converted[name] = try googleOpenAPISchema(
                from: propertySchema,
                isRoot: false,
                referenceContext: referenceContext
            ) ?? .object([:])
        }
        result["properties"] = .object(converted)
    }
    if let items = object["items"] {
        if let array = items.arrayValue {
            result["items"] = .array(try array.map {
                try googleOpenAPISchema(
                    from: $0,
                    isRoot: false,
                    referenceContext: referenceContext
                ) ?? .object([:])
            })
        } else if let converted = try googleOpenAPISchema(
            from: items,
            isRoot: false,
            referenceContext: referenceContext
        ) {
            result["items"] = converted
        }
    }
    for key in ["allOf", "oneOf"] {
        if let array = object[key]?.arrayValue {
            result[key] = .array(try array.map {
                try googleOpenAPISchema(
                    from: $0,
                    isRoot: false,
                    referenceContext: referenceContext
                ) ?? .object([:])
            })
        }
    }
    if let anyOf = object["anyOf"]?.arrayValue {
        let nonNullSchemas = anyOf.filter { $0["type"]?.stringValue != "null" }
        if nonNullSchemas.count != anyOf.count {
            result["nullable"] = true
            if nonNullSchemas.count == 1,
               let converted = try googleOpenAPISchema(
                   from: nonNullSchemas[0],
                   isRoot: false,
                   referenceContext: referenceContext
               )?.objectValue {
                result.merge(converted) { _, new in new }
            } else {
                result["anyOf"] = .array(try nonNullSchemas.map {
                    try googleOpenAPISchema(
                        from: $0,
                        isRoot: false,
                        referenceContext: referenceContext
                    ) ?? .object([:])
                })
            }
        } else {
            result["anyOf"] = .array(try anyOf.map {
                try googleOpenAPISchema(
                    from: $0,
                    isRoot: false,
                    referenceContext: referenceContext
                ) ?? .object([:])
            })
        }
    }

    return result.isEmpty ? nil : .object(result)
}

private func googleOpenAPISchemaReference(
    _ schema: [String: JSONValue],
    reference: String,
    isRoot: Bool,
    referenceContext: GoogleSchemaReferenceContext
) throws -> JSONValue? {
    let resolvedReference = try googleReferencedSchema(
        reference,
        referenceContext: referenceContext
    )
    guard !referenceContext.resolvingReferences.contains(resolvedReference.key) else {
        throw AIError.invalidArgument(
            argument: "schema",
            message: "Google schema conversion does not support recursive JSON Schema references."
        )
    }

    var resolvingReferences = referenceContext.resolvingReferences
    resolvingReferences.insert(resolvedReference.key)

    var siblings = schema
    siblings.removeValue(forKey: "$ref")

    let resolvedSchema: JSONValue
    switch resolvedReference.schema {
    case .bool(true):
        resolvedSchema = .object(siblings)
    case .bool(false):
        resolvedSchema = .bool(false)
    case let .object(definition):
        var resolved = definition
        resolved.merge(siblings) { _, sibling in sibling }
        resolvedSchema = .object(resolved)
    default:
        resolvedSchema = resolvedReference.schema
    }

    return try googleOpenAPISchema(
        from: resolvedSchema,
        isRoot: isRoot,
        referenceContext: GoogleSchemaReferenceContext(
            definitions: referenceContext.definitions,
            dollarDefinitions: referenceContext.dollarDefinitions,
            resolvingReferences: resolvingReferences
        )
    )
}

private func googleReferencedSchema(
    _ reference: String,
    referenceContext: GoogleSchemaReferenceContext
) throws -> (schema: JSONValue, key: String) {
    let prefix: String
    let definitions: [String: JSONValue]?
    if reference.hasPrefix("#/$defs/") {
        prefix = "#/$defs/"
        definitions = referenceContext.dollarDefinitions
    } else if reference.hasPrefix("#/definitions/") {
        prefix = "#/definitions/"
        definitions = referenceContext.definitions
    } else {
        throw googleUnsupportedSchemaReference(reference)
    }

    let encodedName = String(reference.dropFirst(prefix.count))
    guard !encodedName.isEmpty,
          !encodedName.contains("/"),
          let decodedName = encodedName.removingPercentEncoding,
          !decodedName.contains("/"),
          !googleHasInvalidJSONPointerEscape(decodedName),
          let definitions else {
        throw googleUnsupportedSchemaReference(reference)
    }

    let definitionName = decodedName
        .replacingOccurrences(of: "~1", with: "/")
        .replacingOccurrences(of: "~0", with: "~")
    guard let definition = definitions[definitionName] else {
        throw googleUnsupportedSchemaReference(reference)
    }

    return (definition, "\(prefix)\(definitionName)")
}

private func googleHasInvalidJSONPointerEscape(_ value: String) -> Bool {
    var index = value.startIndex
    while let tilde = value[index...].firstIndex(of: "~") {
        let next = value.index(after: tilde)
        guard next != value.endIndex, value[next] == "0" || value[next] == "1" else {
            return true
        }
        index = value.index(after: next)
    }
    return false
}

private func googleUnsupportedSchemaReference(_ reference: String) -> AIError {
    AIError.invalidArgument(
        argument: "schema",
        message: "Google schema conversion only supports references to direct children of root-level $defs or definitions. Unsupported reference: \(reference)"
    )
}

private enum GoogleEnumPrimitiveType: String {
    case string
    case number
    case integer
    case boolean
}

private func googleAddEnumToSchema(
    values: [JSONValue],
    declaredType: JSONValue?,
    result: inout [String: JSONValue]
) throws {
    let declaredTypes = googleDeclaredSchemaTypes(declaredType)
    let nullable = declaredTypes.contains("null") || (declaredType == nil && values.contains(.null))
    let enumValues = nullable ? values.filter { $0 != .null } : values

    if !values.isEmpty, values.allSatisfy({ $0 == .null }) {
        let typeAllowsNull = declaredType == nil || declaredTypes.contains("null")
        if typeAllowsNull {
            result["type"] = .string("null")
            if declaredType?.arrayValue != nil {
                result.removeValue(forKey: "anyOf")
            }
            return
        }
    }

    guard let enumType = googleEnumPrimitiveType(values: enumValues, declaredType: declaredType) else {
        throw AIError.invalidArgument(
            argument: "schema",
            message: "Google does not support this JSON Schema enum. Enum values must share one supported primitive type and match the schema type."
        )
    }

    result["type"] = .string(enumType.rawValue)
    if declaredType?.arrayValue != nil {
        result.removeValue(forKey: "anyOf")
    }
    if nullable {
        result["nullable"] = .bool(true)
    }

    if enumType == .string {
        result["enum"] = .array(enumValues)
    } else {
        result["format"] = .string("enum")
        result["enum"] = .array(enumValues.map { .string(googleEnumWireValue($0)) })
    }
}

private func googleEnumPrimitiveType(values: [JSONValue], declaredType: JSONValue?) -> GoogleEnumPrimitiveType? {
    guard !values.isEmpty else { return nil }

    let allows: (GoogleEnumPrimitiveType) -> Bool = { type in
        guard declaredType != nil else { return true }
        return googleDeclaredSchemaTypes(declaredType).contains(type.rawValue)
    }

    if allows(.string), values.allSatisfy({ $0.stringValue != nil }) {
        return .string
    }

    let numbers = values.compactMap(\.doubleValue)
    if numbers.count == values.count, numbers.allSatisfy(\.isFinite) {
        if allows(.number) {
            return .number
        }
        if allows(.integer), numbers.allSatisfy({ $0.rounded() == $0 }) {
            return .integer
        }
    }

    if allows(.boolean), values.allSatisfy({ $0.boolValue != nil }) {
        return .boolean
    }

    return nil
}

private func googleDeclaredSchemaTypes(_ value: JSONValue?) -> Set<String> {
    if let string = value?.stringValue {
        return [string]
    }
    return Set(value?.arrayValue?.compactMap(\.stringValue) ?? [])
}

private func googleEnumWireValue(_ value: JSONValue) -> String {
    switch value {
    case let .number(number):
        if number.rounded() == number,
           number >= Double(Int.min),
           number <= Double(Int.max) {
            return String(Int(number))
        }
        return String(number)
    case let .bool(boolean):
        return boolean ? "true" : "false"
    default:
        return getErrorMessage(value)
    }
}