import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/candle_model.dart';

// ─────────────────────────────────────────────────
// WS message type enum
// ─────────────────────────────────────────────────
enum WsMessageType {
  tick, emergencyStop, resume, modelReloaded, riskLimitsChanged,
  dataStale, dataResumed, circuitBreaker, agentReportReady,
  backtestProgress, ping, unknown
}

// ─────────────────────────────────────────────────
// WsMessage — typed wrapper around raw JSON
// ─────────────────────────────────────────────────
class WsMessage {
  final WsMessageType type;
  final Map<String, dynamic> raw;

  const WsMessage({required this.type, required this.raw});

  double get price           => (raw['price']           as num?)?.toDouble() ?? 0.0;
  double get portfolioValue  => (raw['portfolio_value'] as num?)?.toDouble() ?? 0.0;
  double get unrealizedPnl   => (raw['unrealized_pnl']  as num?)?.toDouble() ?? 0.0;
  double get drawdown        => (raw['drawdown']         as num?)?.toDouble() ?? 0.0;
  double get action          => (raw['action']           as num?)?.toDouble() ?? 0.0;
  String get hmmRegime       => raw['hmm_regime']?.toString()         ?? 'Unknown';
  double get garchVolatility => (raw['garch_volatility'] as num?)?.toDouble() ?? 0.0;
  int    get tradeCount      => (raw['trade_count']      as int?)    ?? 0;
  String get logMessage      => raw['log_message']?.toString()        ?? '';
  int    get latencyMs       => (raw['latency_ms']       as int?)    ?? 0;
  String get dataQuality     => raw['data_quality']?.toString()       ?? 'good';
  String get agentVerdict    => raw['agent_verdict']?.toString()      ?? 'NEUTRAL';
  int    get agentScore      => (raw['agent_score']      as int?)    ?? 0;
}

// ─────────────────────────────────────────────────
// Candle list notifier — max 500 candles
// ─────────────────────────────────────────────────
class CandleListNotifier extends StateNotifier<List<CandleModel>> {
  CandleListNotifier() : super([]);

  void updateCandle(CandleModel candle) {
    final list = List<CandleModel>.from(state);
    final idx = list.indexWhere((c) => c.timestamp == candle.timestamp);
    if (idx != -1) {
      list[idx] = candle;
    } else {
      list.add(candle);
      if (list.length > 500) list.removeAt(0);
    }
    state = list;
  }
}

// ─────────────────────────────────────────────────
// Unified WebSocket service
// ─────────────────────────────────────────────────
class UnifiedWebSocketService extends StateNotifier<WsMessage?> {
  static const String _baseUrl = 'ws://127.0.0.1:8000/api/v1';

  final Ref _ref;

  // Public getter so providers.dart can watch it
  final CandleListNotifier candleListNotifier = CandleListNotifier();

  WebSocketChannel? _candlesChannel;
  WebSocketChannel? _streamChannel;
  Timer? _reconnectTimer;
  Timer? _pingTimer;
  int _reconnectDelaySec = 2;

  UnifiedWebSocketService(this._ref) : super(null) {
    _connect();
    _startPingTimer();
  }

  // ─── Connection ──────────────────────────────
  void _connect() {
    _candlesChannel = WebSocketChannel.connect(Uri.parse('$_baseUrl/ws/candles'));
    _streamChannel  = WebSocketChannel.connect(Uri.parse('$_baseUrl/stream'));

    _candlesChannel!.stream.listen(
      _handleCandleMessage,
      onError: (_) => _scheduleReconnect(),
      onDone:  ()  => _scheduleReconnect(),
    );

    _streamChannel!.stream.listen(
      _handleStreamMessage,
      onError: (_) => _scheduleReconnect(),
      onDone:  ()  => _scheduleReconnect(),
    );
  }

  // ─── Handlers ────────────────────────────────
  void _handleCandleMessage(dynamic raw) {
    try {
      final data   = json.decode(raw as String) as Map<String, dynamic>;
      final candle = CandleModel.fromJson(data);
      candleListNotifier.updateCandle(candle);
      _reconnectDelaySec = 2; // Reset backoff on success
    } catch (_) {}
  }

  void _handleStreamMessage(dynamic raw) {
    try {
      final data    = json.decode(raw as String) as Map<String, dynamic>;
      final typeStr = data['type']?.toString().toLowerCase() ?? '';
      final type    = _parseType(typeStr);
      state = WsMessage(type: type, raw: data);
      _reconnectDelaySec = 2; // Reset backoff on success
    } catch (_) {}
  }

  WsMessageType _parseType(String s) {
    switch (s) {
      case 'tick':                return WsMessageType.tick;
      case 'emergency_stop':      return WsMessageType.emergencyStop;
      case 'resume':              return WsMessageType.resume;
      case 'model_reloaded':      return WsMessageType.modelReloaded;
      case 'risk_limits_changed': return WsMessageType.riskLimitsChanged;
      case 'data_stale':          return WsMessageType.dataStale;
      case 'data_resumed':        return WsMessageType.dataResumed;
      case 'circuit_breaker':     return WsMessageType.circuitBreaker;
      case 'agent_report_ready':  return WsMessageType.agentReportReady;
      case 'backtest_progress':   return WsMessageType.backtestProgress;
      case 'ping':                return WsMessageType.ping;
      default:                    return WsMessageType.unknown;
    }
  }

  // ─── Reconnect with exponential backoff ──────
  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(Duration(seconds: _reconnectDelaySec), () {
      _reconnectDelaySec = (_reconnectDelaySec * 2).clamp(2, 30);
      _connect();
    });
  }

  // ─── Keep-alive ping ─────────────────────────
  void _startPingTimer() {
    _pingTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      final ping = json.encode({'type': 'ping'});
      try { _candlesChannel?.sink.add(ping); } catch (_) {}
      try { _streamChannel?.sink.add(ping);  } catch (_) {}
    });
  }

  @override
  void dispose() {
    _reconnectTimer?.cancel();
    _pingTimer?.cancel();
    _candlesChannel?.sink.close();
    _streamChannel?.sink.close();
    candleListNotifier.dispose();
    super.dispose();
  }
}
