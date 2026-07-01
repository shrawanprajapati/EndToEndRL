import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'models/candle_model.dart';
import 'models/chat_message.dart';
import 'models/trade_model.dart';
import 'models/news_model.dart';
import 'services/api_service.dart';
import 'services/websocket_service.dart';

// ─────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────
final isDarkModeProvider = StateProvider<bool>((ref) => true);

// ─────────────────────────────────────────────────
// WebSocket — single instance, drives all live data
// ─────────────────────────────────────────────────
final wsServiceProvider = StateNotifierProvider<UnifiedWebSocketService, WsMessage?>((ref) {
  return UnifiedWebSocketService(ref);
});

// ─────────────────────────────────────────────────
// Candles — separate notifier exposed by wsService
// ─────────────────────────────────────────────────
final candlesProvider = StateNotifierProvider<CandleListNotifier, List<CandleModel>>((ref) {
  // Access the notifier from wsServiceProvider to get the same CandleListNotifier instance
  return ref.watch(wsServiceProvider.notifier).candleListNotifier;
});

// ─────────────────────────────────────────────────
// REST API — auto-refresh every 30 s
// ─────────────────────────────────────────────────
final regimeProvider = StreamProvider<Map<String, dynamic>>((ref) async* {
  while (true) {
    try {
      yield await ApiService.getRegime();
    } catch (_) {
      // Silently skip; UI handles empty data state
    }
    await Future.delayed(const Duration(seconds: 30));
  }
});

final backtestProvider = FutureProvider<Map<String, dynamic>>((ref) => ApiService.getBacktest());
final featureImportanceProvider = FutureProvider<List<Map<String, dynamic>>>((ref) => ApiService.getFeatureImportance());
final latestReportProvider = FutureProvider<Map<String, dynamic>>((ref) => ApiService.getLatestReport());
final newsProvider = FutureProvider<List<NewsModel>>((ref) async {
  final newsData = await ApiService.getNews();
  return newsData.map((json) => NewsModel.fromJson(json)).toList();
});

// ─────────────────────────────────────────────────
// Indicator selection (market chart)
// ─────────────────────────────────────────────────
final selectedIndicatorsProvider = StateProvider<Set<String>>((ref) => {'EMA20', 'RSI'});

// ─────────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────────
final chatSessionIdProvider = StateProvider<String>((ref) => DateTime.now().millisecondsSinceEpoch.toString());

class ChatMessagesNotifier extends StateNotifier<List<ChatMessage>> {
  ChatMessagesNotifier() : super([]);

  void addMessage(ChatMessage msg) {
    final updated = [...state, msg];
    state = updated.length > 50 ? updated.sublist(updated.length - 50) : updated;
  }

  void clear() => state = [];
}

final chatMessagesProvider = StateNotifierProvider<ChatMessagesNotifier, List<ChatMessage>>(
  (ref) => ChatMessagesNotifier(),
);

// ─────────────────────────────────────────────────
// Backtest date range
// ─────────────────────────────────────────────────
final backtestFromDateProvider = StateProvider<DateTime>(
  (ref) => DateTime.now().subtract(const Duration(days: 180)),
);
final backtestToDateProvider = StateProvider<DateTime>((ref) => DateTime.now());

// ─────────────────────────────────────────────────
// Binance credentials  (in-memory ONLY — never persisted)
// ─────────────────────────────────────────────────
final binanceApiKeyProvider = StateProvider<String>((ref) => '');
final binanceSecretProvider = StateProvider<String>((ref) => '');
final binanceConnectedProvider = StateProvider<bool>((ref) => false);

// ─────────────────────────────────────────────────
// Virtual portfolio simulation
// ─────────────────────────────────────────────────
class VirtualPortfolioState {
  final double portfolioValue;
  final double btcAllocation;
  final List<TradeModel> trades;
  final double totalPnl;
  final double maxDrawdown;
  final double currentDrawdown;

  const VirtualPortfolioState({
    required this.portfolioValue,
    required this.btcAllocation,
    required this.trades,
    required this.totalPnl,
    required this.maxDrawdown,
    required this.currentDrawdown,
  });

  static const initial = VirtualPortfolioState(
    portfolioValue: 10000.0,
    btcAllocation: 0.0,
    trades: [],
    totalPnl: 0.0,
    maxDrawdown: 0.0,
    currentDrawdown: 0.0,
  );
}

class VirtualPortfolioNotifier extends StateNotifier<VirtualPortfolioState> {
  VirtualPortfolioNotifier() : super(VirtualPortfolioState.initial);

  void updateFromWsMessage(WsMessage msg) {
    // Listen for engine ticker iterations safely
    if (msg.type != WsMessageType.tick && msg.raw['type'] != 'tick') return;
    
    final newMax = msg.drawdown > state.maxDrawdown ? msg.drawdown : state.maxDrawdown;
    
    // Append program historical trace points dynamically 
    final currentTrade = TradeModel(
      timestamp: DateTime.now(),
      side: msg.agentVerdict,
      price: msg.price,
      allocation: msg.action,
      portfolioValue: msg.portfolioValue > 0 ? msg.portfolioValue : state.portfolioValue,
      pnl: msg.unrealizedPnl,
      drawdown: msg.drawdown,
    );

    state = VirtualPortfolioState(
      portfolioValue: msg.portfolioValue > 0 ? msg.portfolioValue : state.portfolioValue,
      btcAllocation: msg.action,
      trades: [...state.trades, currentTrade].extraThrottle(50), // keep last 50 points
      totalPnl: msg.unrealizedPnl,
      maxDrawdown: newMax,
      currentDrawdown: msg.drawdown,
    );
  }

  void reset() => state = VirtualPortfolioState.initial;
}

// Inline helper to prevent array expansion crashes
extension on List<TradeModel> {
  List<TradeModel> extraThrottle(int max) => length > max ? sublist(length - max) : this;
}

final virtualPortfolioProvider = StateNotifierProvider<VirtualPortfolioNotifier, VirtualPortfolioState>((ref) {
  final notifier = VirtualPortfolioNotifier();
  ref.listen<WsMessage?>(wsServiceProvider, (_, next) {
    if (next != null) notifier.updateFromWsMessage(next);
  });
  return notifier;
});
