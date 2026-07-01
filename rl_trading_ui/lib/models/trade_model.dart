class TradeModel {
  final DateTime timestamp;
  final String side; // "BUY" | "SELL" | "HOLD"
  final double price;
  final double allocation;
  final double portfolioValue;
  final double pnl;
  final double drawdown;

  TradeModel({
    required this.timestamp,
    required this.side,
    required this.price,
    required this.allocation,
    required this.portfolioValue,
    required this.pnl,
    required this.drawdown,
  });

  factory TradeModel.fromJson(Map<String, dynamic> json) {
    return TradeModel(
      timestamp: DateTime.parse(json['timestamp']),
      side: json['side'] ?? 'HOLD',
      price: (json['price'] as num).toDouble(),
      allocation: (json['allocation'] as num).toDouble(),
      portfolioValue: (json['portfolio_value'] as num).toDouble(),
      pnl: (json['pnl'] as num).toDouble(),
      drawdown: (json['drawdown'] as num).toDouble(),
    );
  }
}
