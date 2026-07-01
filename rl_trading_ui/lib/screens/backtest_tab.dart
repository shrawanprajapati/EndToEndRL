import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import '../providers.dart';
import '../services/api_service.dart';
import '../app_theme.dart';
import '../widgets/metric_card.dart';

class BacktestTab extends ConsumerStatefulWidget {
  const BacktestTab({super.key});
  @override
  ConsumerState<BacktestTab> createState() => _BacktestTabState();
}

class _BacktestTabState extends ConsumerState<BacktestTab> {
  String _activeMode = 'Virtual';
  String _datasetMode = 'test';
  String _strategy = 'all';
  bool _isAccepted = false;
  Map<String, dynamic>? _strategyBacktest;
  bool _isRunning = false;
  String? _error;


  static const List<String> _datasetModes = ['train', 'test', 'full'];
  static const List<String> _strategies = ['all', 'sma', 'ema', 'rsi', 'macd', 'bollinger', 'buy_hold','rl_model'];

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'Virtual', label: Text('Virtual Backtest'), icon: Icon(Icons.history)),
              ButtonSegment(value: 'Live', label: Text('Live Deploy'), icon: Icon(Icons.account_balance_wallet)),
            ],
            selected: {_activeMode},
            onSelectionChanged: (set) => setState(() => _activeMode = set.first),
          ),
        ),
        Expanded(
          child: _activeMode == 'Virtual' ? _buildVirtualMode() : _buildLiveMode(),
        ),
      ],
    );
  }

  Widget _buildVirtualMode() {
    final fromDate = ref.watch(backtestFromDateProvider);
    final toDate = ref.watch(backtestToDateProvider);
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final d = await showDatePicker(context: context, initialDate: fromDate, firstDate: DateTime(2020), lastDate: DateTime.now());
                    if (d != null) ref.read(backtestFromDateProvider.notifier).state = d;
                  },
                  icon: const Icon(Icons.calendar_today, size: 16),
                  label: Text('From: ${DateFormat.yMd().format(fromDate)}'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final d = await showDatePicker(context: context, initialDate: toDate, firstDate: DateTime(2020), lastDate: DateTime.now());
                    if (d != null) ref.read(backtestToDateProvider.notifier).state = d;
                  },
                  icon: const Icon(Icons.calendar_today, size: 16),
                  label: Text('To: ${DateFormat.yMd().format(toDate)}'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _datasetMode,
                  decoration: const InputDecoration(labelText: 'Dataset'),
                  items: _datasetModes
                      .map((mode) => DropdownMenuItem(value: mode, child: Text(mode.toUpperCase())))
                      .toList(),
                  onChanged: _isRunning ? null : (value) => setState(() => _datasetMode = value ?? _datasetMode),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _strategy,
                  decoration: const InputDecoration(labelText: 'Strategy'),
                  items: _strategies
                      .map((strategy) => DropdownMenuItem(value: strategy, child: Text(strategy.replaceAll('_', ' ').toUpperCase())))
                      .toList(),
                  onChanged: _isRunning ? null : (value) => setState(() => _strategy = value ?? _strategy),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _isRunning ? null : _runStrategyBacktest,
            style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primary, foregroundColor: Colors.white, minimumSize: const Size(double.infinity, 50)),
            icon: _isRunning
                ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.play_arrow),
            label: Text(_isRunning ? 'RUNNING...' : 'RUN BACKTEST'),
          ),
          const SizedBox(height: 24),
          _buildBacktestResults(),
        ],
      ),
    );
  }

  Widget _buildBacktestResults() {
    if (_error != null) {
      return Text(_error!, style: const TextStyle(color: AppTheme.red));
    }

    final rows = List<Map<String, dynamic>>.from(_strategyBacktest?['comparison'] ?? const []);
    if (rows.isEmpty) {
      return const Text('No strategy backtest has been run yet.', style: TextStyle(color: Colors.grey));
    }

    final row = rows.firstWhere(
      (item) => item['Strategy'] == _strategy,
      orElse: () => rows.firstWhere((item) => item['Strategy'] == 'rl_model', orElse: () => rows.first),
    );

    return Column(
      children: [
        Row(
          children: [
            Expanded(child: MetricCard(label: 'TOTAL RETURN', value: _formatPercent(row['Total Return [%]']), color: _metricColor(row['Total Return [%]']))),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'SHARPE', value: _formatNumber(row['Sharpe Ratio']), color: _metricColor(row['Sharpe Ratio']))),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'CALMAR', value: _formatNumber(row['Calmar Ratio']), color: _metricColor(row['Calmar Ratio']))),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(child: MetricCard(label: 'MAX DRAWDOWN', value: _formatDrawdown(row['Max Drawdown [%]']), color: AppTheme.amber)),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'WIN RATE', value: _formatPercent(row['Win Rate [%]']))),
            const SizedBox(width: 8),
            Expanded(child: MetricCard(label: 'TRADES', value: _formatNumber(row['# Trades']))),
          ],
        ),
        const SizedBox(height: 16),
        ...rows.map((item) => ListTile(
              dense: true,
              contentPadding: EdgeInsets.zero,
              title: Text('${item['Strategy']}'.replaceAll('_', ' ').toUpperCase()),
              subtitle: Text('Return ${_formatPercent(item['Total Return [%]'])} | Sharpe ${_formatNumber(item['Sharpe Ratio'])}'),
              trailing: Text(_formatMoney(item['Final Value'])),
            )),
      ],
    );
  }

  // Update this function in lib/screens/backtest_tab.dart
  Future<void> _runStrategyBacktest() async {
    setState(() {
      _isRunning = true;
      _error = null;
    });
    try {
      // 1. Ensure you are sending the actual state variables
      // Replace the generic ApiService call with this:
      final result = await ApiService.runVectorbtBacktest(
        mode: _datasetMode,    // Sends 'train', 'test', or 'full'
        strategy: _strategy    // Sends 'sma', 'rl_model', etc.
      );
      
      if (!mounted) return;
      setState(() => _strategyBacktest = result);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _isRunning = false);
    }
  }

  Color _metricColor(dynamic value) {
    final n = _asDouble(value);
    if (n == null) return Colors.grey;
    return n >= 0 ? AppTheme.green : AppTheme.red;
  }

  String _formatPercent(dynamic value) {
    final n = _asDouble(value);
    if (n == null) return '--';
    final sign = n > 0 ? '+' : '';
    return '$sign${n.toStringAsFixed(2)}%';
  }

  String _formatDrawdown(dynamic value) {
    final n = _asDouble(value);
    if (n == null) return '--';
    return '${n.toStringAsFixed(2)}%';
  }

  String _formatNumber(dynamic value) {
    final n = _asDouble(value);
    if (n == null) return '--';
    return n.toStringAsFixed(n.abs() >= 100 ? 0 : 2);
  }

  String _formatMoney(dynamic value) {
    final n = _asDouble(value);
    if (n == null) return '--';
    return NumberFormat.compactCurrency(symbol: '\$').format(n);
  }

  double? _asDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    return double.tryParse(value.toString());
  }

  Widget _buildLiveMode() {
    if (!_isAccepted) {
      return Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.warning_amber_rounded, color: AppTheme.red, size: 64),
            const SizedBox(height: 16),
            const Text('REAL TRADING MODE', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: AppTheme.red)),
            const SizedBox(height: 12),
            const Text(
              'Orders execute with real funds. All trades are irreversible. QuantPilot AI is experimental software.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 24),
            CheckboxListTile(
              title: const Text('I understand this uses real money'),
              value: _isAccepted,
              onChanged: (val) => setState(() => _isAccepted = val!),
              controlAffinity: ListTileControlAffinity.leading,
            ),
          ],
        ),
      );
    }
    return _buildBinanceInterface();
  }

  Widget _buildBinanceInterface() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          const TextField(decoration: InputDecoration(labelText: 'Binance API Key')),
          const TextField(decoration: InputDecoration(labelText: 'Binance Secret'), obscureText: true),
          const SizedBox(height: 16),
          ElevatedButton(onPressed: () {}, child: const Text('TEST CONNECTION')),
        ],
      ),
    );
  }
}
