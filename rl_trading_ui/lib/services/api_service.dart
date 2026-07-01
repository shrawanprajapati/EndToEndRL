import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}

class ApiService {
  static const String baseUrl = "http://127.0.0.1:8000";
  static const String apiSecret = String.fromEnvironment('API_SECRET', defaultValue: 'changeme');

  static Map<String, String> get _headers => {
    "Content-Type": "application/json",
    "x-api-secret": apiSecret,
  };

  static Future<T> _request<T>(Future<http.Response> Function() call, T Function(dynamic body) parser) async {
    try {
      final response = await call().timeout(const Duration(seconds: 100));
      final dynamic body = json.decode(response.body);
      
      if (response.statusCode == 200) {
        return parser(body);
      } else {
        throw ApiException("HTTP ${response.statusCode}: ${response.body}");
      }
    } on http.ClientException {
      throw ApiException("Network error occurred");
    } catch (e) {
      if (e.toString().contains("timeout")) {
        throw ApiException("Request timed out");
      }
      rethrow;
    }
  }

  static Future<Map<String, dynamic>> getRegime() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/regime"), headers: _headers), (b) => b as Map<String, dynamic>);

  static Future<Map<String, dynamic>> getBacktest() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/backtest"), headers: _headers), (b) => b as Map<String, dynamic>);

  static Future<Map<String, dynamic>> getHealth() => 
    _request(() => http.get(Uri.parse("$baseUrl/health")), (b) => b as Map<String, dynamic>);

  static Future<List<Map<String, dynamic>>> getCheckpoints() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/checkpoints"), headers: _headers), (b) => List<Map<String, dynamic>>.from(b['checkpoints'] ?? []));

  static Future<Map<String, dynamic>> getLatestReport() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/agent/report/latest"), headers: _headers), (b) => b as Map<String, dynamic>);

  static Future<List<Map<String, dynamic>>> getReportHistory() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/agent/report/history"), headers: _headers), (b) => List<Map<String, dynamic>>.from(b['history'] ?? []));

  static Future<List<Map<String, dynamic>>> getFeatureImportance() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/feature_importance"), headers: _headers), (b) => List<Map<String, dynamic>>.from(b['importance'] ?? []));

  static Future<List<Map<String, dynamic>>> getNews() => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/news"), headers: _headers), (b) => List<Map<String, dynamic>>.from(b['news'] ?? []));

  static Future<List<Map<String, dynamic>>> getLatestFeatures({int n = 5}) => 
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/features/latest?n=$n"), headers: _headers), (b) => List<Map<String, dynamic>>.from(b));

  static Future<Map<String, dynamic>> sendChat({required String message, required String sessionId}) => 
    _request(() => http.post(
      Uri.parse("$baseUrl/api/v1/agent/chat"),
      headers: _headers,
      body: json.encode({"message": message, "session_id": sessionId}),
    ), (b) => b as Map<String, dynamic>);

  static Future<void> resetChat({required String sessionId}) => 
    _request(() => http.post(
      Uri.parse("$baseUrl/api/v1/agent/chat/reset"),
      headers: _headers,
      body: json.encode({"session_id": sessionId}),
    ), (b) => null);

  static Future<Map<String, dynamic>> triggerReport() => 
    _request(() => http.post(Uri.parse("$baseUrl/api/v1/agent/report"), headers: _headers), (b) => b as Map<String, dynamic>);

  static Future<Map<String, dynamic>> runBacktest({required DateTime fromDate, required DateTime toDate}) => 
    _request(() => http.post(
      Uri.parse("$baseUrl/api/v1/run_backtest"),
      headers: _headers,
      body: json.encode({"from": fromDate.toIso8601String(), "to": toDate.toIso8601String()}),
    ), (b) => b as Map<String, dynamic>);

  static Future<Map<String, dynamic>> getBacktestingStrategies() =>
    _request(() => http.get(Uri.parse("$baseUrl/api/v1/backtesting/strategies"), headers: _headers), (b) => b as Map<String, dynamic>);

  static Future<Map<String, dynamic>> runStrategyBacktest({required String mode, required String strategy}) =>
    _request(() => http.post(
      Uri.parse("$baseUrl/api/v1/backtesting/run"),
      headers: _headers,
      body: json.encode({"mode": mode, "strategy": strategy}),
    ), (b) => b as Map<String, dynamic>);
  
  // Add this to your ApiService class in lib/services/api_service.dart
  static Future<Map<String, dynamic>> runVectorbtBacktest({
    required String mode, 
    required String strategy
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/v1/run_backtest'),
      headers: {'Content-Type': 'application/json', 'x-api-secret': 'changeme'},
      body: json.encode({'mode': mode, 'strategy': strategy}),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to run backtest: ${response.statusCode}');
    }
  }
}
