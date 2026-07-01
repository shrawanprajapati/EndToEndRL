import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:intl/intl.dart';
import '../models/candle_model.dart';
import '../providers.dart';
import '../widgets/metric_card.dart';
import '../app_theme.dart';
import '../services/websocket_service.dart';

class MarketTab extends ConsumerWidget {
  const MarketTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final candles = ref.watch(candlesProvider);
    final wsMessage = ref.watch(wsServiceProvider);
    final regime = ref.watch(regimeProvider).asData?.value;
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(context, wsMessage, regime),
          const SizedBox(height: 24),
          _buildIndicatorSelector(ref),
          const SizedBox(height: 16),
          _buildChart(context, ref, candles),
          const SizedBox(height: 24),
          _buildRiskPanel(context, wsMessage, regime),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context, WsMessage? ws, Map<String, dynamic>? regime) {
    final price = ws?.price ?? 0.0;
    final formattedPrice = NumberFormat.currency(symbol: '\$', decimalDigits: 2).format(price);
    final regimeLabel = regime?['hmm_regime_label'] ?? 'Unknown';
    final vol = regime?['garch_volatility'] ?? 0.0;

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(formattedPrice, style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold, color: AppTheme.verdictColor(regimeLabel))),
            const Text('24h change: +2.45%', style: TextStyle(color: AppTheme.green, fontSize: 14)),
          ],
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppTheme.verdictColor(regimeLabel).withOpacity(0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(regimeLabel, style: TextStyle(color: AppTheme.verdictColor(regimeLabel), fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 4),
            Text('Vol: ${vol.toStringAsFixed(4)}', style: const TextStyle(color: Colors.grey, fontSize: 12)),
          ],
        )
      ],
    );
  }

  Widget _buildIndicatorSelector(WidgetRef ref) {
    final selected = ref.watch(selectedIndicatorsProvider);
    final List<String> indicators = ['EMA20', 'EMA50', 'SMA200', 'RSI', 'MACD', 'BollingerBands', 'VWAP', 'ATR', 'Volume', 'StochRSI'];
    
    return SizedBox(
      height: 40,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: indicators.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final indicator = indicators[index];
          final isSelected = selected.contains(indicator);
          return FilterChip(
            label: Text(indicator, style: TextStyle(fontSize: 12, color: isSelected ? Colors.white : Colors.grey)),
            selected: isSelected,
            onSelected: (val) {
              final newSet = Set<String>.from(selected);
              if (val) {
                newSet.add(indicator);
              } else {
                newSet.remove(indicator);
              }
              ref.read(selectedIndicatorsProvider.notifier).state = newSet;
            },
            selectedColor: AppTheme.primary,
            backgroundColor: Theme.of(context).cardColor,
          );
        },
      ),
    );
  }

  Widget _buildChart(BuildContext context, WidgetRef ref, List<CandleModel> candles) {
    final selected = ref.watch(selectedIndicatorsProvider);
    final List<double> prices = candles.map((e) => e.close).toList();
    
    return Column(
      children: [
        Container(
          height: 320,
          decoration: BoxDecoration(
            color: const Color(0xFF0D0D14),
            borderRadius: BorderRadius.circular(12),
          ),
          child: SfCartesianChart(
            trackballBehavior: TrackballBehavior(enable: true, activationMode: ActivationMode.singleTap),
            zoomPanBehavior: ZoomPanBehavior(enablePanning: true, enablePinching: true),
            primaryXAxis: DateTimeAxis(dateFormat: DateFormat.Hm()),
            primaryYAxis: NumericAxis(opposedPosition: true),
            series: <CartesianSeries>[
              CandleSeries<CandleModel, DateTime>(
                dataSource: candles,
                xValueMapper: (c, _) => c.timestamp,
                lowValueMapper: (c, _) => c.low,
                highValueMapper: (c, _) => c.high,
                openValueMapper: (c, _) => c.open,
                closeValueMapper: (c, _) => c.close,
                bearColor: AppTheme.red,
                bullColor: AppTheme.green,
              ),
              if (selected.contains('EMA20')) ...[
                LineSeries<double, DateTime>(
                  dataSource: _computeEMA(prices, 20),
                  xValueMapper: (_, i) => candles[i].timestamp,
                  yValueMapper: (v, _) => v,
                  color: Colors.orange,
                ),
              ],
              if (selected.contains('EMA50')) ...[
                LineSeries<double, DateTime>(
                  dataSource: _computeEMA(prices, 50),
                  xValueMapper: (_, i) => candles[i].timestamp,
                  yValueMapper: (v, _) => v,
                  color: Colors.blue,
                ),
              ],
              if (selected.contains('SMA200')) ...[
                LineSeries<double, DateTime>(
                  dataSource: _computeSMA(prices, 200),
                  xValueMapper: (_, i) => candles[i].timestamp,
                  yValueMapper: (v, _) => v,
                  color: Colors.purple,
                ),
              ],
            ],
          ),
        ),
        if (selected.contains('RSI')) _buildSubChart('RSI', candles, _computeRSI(prices, 14)),
        if (selected.contains('MACD')) _buildSubChart('MACD', candles, _computeMACD(prices)['histogram']!),
      ],
    );
  }

  Widget _buildSubChart(String name, List<CandleModel> candles, List<double> data) {
    if (data.length != candles.length) return const SizedBox.shrink();
    return Container(
      height: 120,
      margin: const EdgeInsets.only(top: 8),
      decoration: BoxDecoration(color: const Color(0xFF0D0D14), borderRadius: BorderRadius.circular(12)),
      child: SfCartesianChart(
        primaryXAxis: DateTimeAxis(isVisible: false),
        primaryYAxis: NumericAxis(opposedPosition: true),
        series: <CartesianSeries>[
          LineSeries<double, DateTime>(
            dataSource: data,
            xValueMapper: (_, i) => candles[i].timestamp,
            yValueMapper: (v, _) => v,
            color: AppTheme.primary,
          )
        ],
      ),
    );
  }

  Widget _buildRiskPanel(BuildContext context, WsMessage? ws, Map<String, dynamic>? regime) {
    final dd = regime?['drawdown_72h'] ?? 0.0;
    final vol = regime?['garch_volatility'] ?? 0.0;
    
    return Column(
      children: [
        Row(
          children: [
            Expanded(child: MetricCard(label: 'DRAWDOWN', value: '${(dd * 100).toStringAsFixed(2)}%', color: dd > 0.15 ? AppTheme.red : AppTheme.green)),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'VOLATILITY', value: vol > 0.02 ? 'HIGH' : 'LOW', color: vol > 0.02 ? AppTheme.red : AppTheme.green)),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'MAX EXPOSURE', value: '80%')),
          ],
        ),
        const SizedBox(height: 16),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: dd > 0.15 ? AppTheme.red.withOpacity(0.1) : AppTheme.green.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: dd > 0.15 ? AppTheme.red.withOpacity(0.3) : AppTheme.green.withOpacity(0.3)),
          ),
          child: Row(
            children: [
              Icon(dd > 0.15 ? Icons.warning : Icons.check_circle, color: dd > 0.15 ? AppTheme.red : AppTheme.green, size: 16),
              const SizedBox(width: 8),
              Text(dd > 0.15 ? 'CRITICAL DRAWDOWN' : 'System Normal', style: TextStyle(color: dd > 0.15 ? AppTheme.red : AppTheme.green, fontWeight: FontWeight.bold, fontSize: 12)),
            ],
          ),
        )
      ],
    );
  }

  // Indicator Compute Helpers
  List<double> _computeEMA(List<double> prices, int period) {
    if (prices.length < period) return List.filled(prices.length, 0.0);
    List<double> ema = List.filled(prices.length, 0.0);
    double k = 2 / (period + 1);
    ema[0] = prices[0];
    for (int i = 1; i < prices.length; i++) {
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k);
    }
    return ema;
  }

  List<double> _computeSMA(List<double> prices, int period) {
    List<double> sma = List.filled(prices.length, 0.0);
    for (int i = period - 1; i < prices.length; i++) {
        double sum = 0;
        for (int j = 0; j < period; j++) {
            sum += prices[i - j];
        }
        sma[i] = sum / period;
    }
    return sma;
  }

  List<double> _computeRSI(List<double> prices, int period) {
    if (prices.length <= period) return List.filled(prices.length, 50.0);
    List<double> rsi = List.filled(prices.length, 50.0);
    // Logic for RSI computation
    return rsi;
  }

  Map<String, List<double>> _computeMACD(List<double> prices) {
    List<double> ema12 = _computeEMA(prices, 12);
    List<double> ema26 = _computeEMA(prices, 26);
    List<double> macdLine = List.generate(prices.length, (i) => ema12[i] - ema26[i]);
    List<double> signalLine = _computeEMA(macdLine, 9);
    List<double> histogram = List.generate(prices.length, (i) => macdLine[i] - signalLine[i]);
    return {'macd': macdLine, 'signal': signalLine, 'histogram': histogram};
  }
}
