// WARNING: This service executes REAL trades with REAL money.
// All orders are irreversible. Use testConnectivity() before 
// any trade. Never store API keys in source code.

import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

class BinanceService {
  final String apiKey;
  final String apiSecret;

  BinanceService({required this.apiKey, required this.apiSecret});

  Future<bool> testConnectivity() async {
    try {
      final response = await http.get(Uri.parse("https://api.binance.com/api/v3/ping"));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<Map<String, dynamic>> getAccountInfo() async {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final queryString = "timestamp=$timestamp";
    final signature = _sign(queryString, apiSecret);
    
    final response = await http.get(
      Uri.parse("https://api.binance.com/api/v3/account?$queryString&signature=$signature"),
      headers: {"X-MBX-APIKEY": apiKey},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Binance API Error ${response.statusCode}: ${response.body}");
    }
  }

  Future<Map<String, dynamic>> placeBtcOrder({
    required String side, // "BUY" | "SELL"
    required double quantity,
    String type = "MARKET",
  }) async {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final queryString = "symbol=BTCUSDT&side=$side&type=$type&quantity=$quantity&timestamp=$timestamp";
    final signature = _sign(queryString, apiSecret);

    final response = await http.post(
      Uri.parse("https://api.binance.com/api/v3/order?$queryString&signature=$signature"),
      headers: {"X-MBX-APIKEY": apiKey},
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Binance Order Error ${response.statusCode}: ${response.body}");
    }
  }

  Future<double> getBtcUsdtPrice() async {
    final response = await http.get(Uri.parse("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return double.parse(data['price']);
    } else {
      throw Exception("Failed to fetch BTC price");
    }
  }

  static String _sign(String queryString, String secret) {
    final key = utf8.encode(secret);
    final bytes = utf8.encode(queryString);
    final hmacSha256 = Hmac(sha256, key);
    final digest = hmacSha256.convert(bytes);
    return digest.toString();
  }
}
