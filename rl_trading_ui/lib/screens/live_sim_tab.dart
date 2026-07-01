import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:fl_chart/fl_chart.dart';
import '../providers.dart';
import '../widgets/metric_card.dart';
import '../widgets/glass_card.dart';
import '../app_theme.dart';
import '../services/websocket_service.dart';

class LiveSimTab extends ConsumerWidget {
  const LiveSimTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final wsMessage = ref.watch(wsServiceProvider);
    final portfolio = ref.watch(virtualPortfolioProvider);
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _buildDecisionCard(context, wsMessage),
          const SizedBox(height: 16),
          _buildPortfolioCard(context, portfolio),
          const SizedBox(height: 16),
          _buildEquityChart(context, portfolio),
          const SizedBox(height: 16),
          _buildMetricsRow(context, wsMessage, portfolio),
          const SizedBox(height: 16),
          _buildTerminal(context, wsMessage),
        ],
      ),
    );
  }

  Widget _buildDecisionCard(BuildContext context, WsMessage? ws) {
    final action = (ws?.action ?? 0.0) * 100;
    return GlassCard(
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('MODEL DECISION', style: TextStyle(color: Colors.grey, fontSize: 10)),
              Text('Allocation: ${action.toStringAsFixed(1)}%', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
            ],
          ),
          Icon(
            action > 50 ? Icons.trending_up : Icons.trending_flat,
            color: action > 50 ? AppTheme.green : Colors.grey,
            size: 40,
          ),
        ],
      ),
    );
  }

  Widget _buildPortfolioCard(BuildContext context, VirtualPortfolioState portfolio) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Theme.of(context).dividerColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('VIRTUAL PORTFOLIO', style: TextStyle(color: Colors.grey, fontSize: 12)),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('\$${portfolio.portfolioValue.toStringAsFixed(2)}', style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
              Text(
                '${portfolio.totalPnl > 0 ? '+' : ''}${portfolio.totalPnl.toStringAsFixed(2)}%',
                style: TextStyle(color: portfolio.totalPnl >= 0 ? AppTheme.green : AppTheme.red, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: portfolio.btcAllocation,
              minHeight: 8,
              backgroundColor: Colors.white10,
              valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primary),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEquityChart(BuildContext context, VirtualPortfolioState portfolio) {
    // Generate programmatic spots from historical trade tracking instead of hardcoded numbers
    List<FlSpot> spots = [];
    if (portfolio.trades.isEmpty) {
      spots = [const FlSpot(0, 10000), const FlSpot(1, 10000)];
    } else {
      for (int i = 0; i < portfolio.trades.length; i++) {
        spots.add(FlSpot(i.toDouble(), portfolio.trades[i].portfolioValue));
      }
    }

    return Container(
      height: 200,
      padding: const EdgeInsets.only(top: 24, right: 16, left: 16, bottom: 8),
      decoration: BoxDecoration(color: Theme.of(context).cardColor, borderRadius: BorderRadius.circular(16)),
      child: LineChart(
        LineChartData(
          gridData: const FlGridData(show: false),
          titlesData: const FlTitlesData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: AppTheme.primary,
              barWidth: 3,
              dotData: const FlDotData(show: false),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricsRow(BuildContext context, WsMessage? ws, VirtualPortfolioState portfolio) {
    return Row(
      children: [
        Expanded(child: MetricCard(label: 'DRAWDOWN', value: '${(portfolio.currentDrawdown * 100).toStringAsFixed(2)}%')),
        const SizedBox(width: 8),
        Expanded(child: MetricCard(label: 'TRADES', value: '${ws?.tradeCount ?? portfolio.trades.length}')),
        const SizedBox(width: 8),
        Expanded(child: MetricCard(label: 'LATENCY', value: '${ws?.latencyMs ?? 0}ms')),
      ],
    );
  }

  Widget _buildTerminal(BuildContext context, WsMessage? ws) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: Colors.black, borderRadius: BorderRadius.circular(8)),
      child: Text(ws?.logMessage ?? 'Awaiting stream...', style: const TextStyle(color: Colors.green, fontSize: 10, fontFamily: 'monospace')),
    );
  }
}
