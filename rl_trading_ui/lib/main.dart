import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'app_theme.dart';
import 'providers.dart';
import 'screens/market_tab.dart';
import 'screens/live_sim_tab.dart';
import 'screens/backtest_tab.dart';
import 'screens/news_tab.dart';
import 'screens/chat_tab.dart';

void main() {
  runApp(const ProviderScope(child: QuantPilotApp()));
}

class QuantPilotApp extends ConsumerWidget {
  const QuantPilotApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isDarkMode = ref.watch(isDarkModeProvider);
    return MaterialApp(
      title: 'QuantPilot AI',
      debugShowCheckedModeBanner: false,
      theme: isDarkMode ? AppTheme.darkTheme : AppTheme.lightTheme,
      home: const MainShell(),
    );
  }
}

class MainShell extends ConsumerStatefulWidget {
  const MainShell({super.key});
  @override
  ConsumerState<MainShell> createState() => _MainShellState();
}

class _MainShellState extends ConsumerState<MainShell> {
  final PageController _pageController = PageController();
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('QuantPilot AI', style: TextStyle(fontWeight: FontWeight.bold)),
        actions: [
          const PriceBadge(),
          IconButton(
            icon: Icon(ref.watch(isDarkModeProvider) ? Icons.light_mode : Icons.dark_mode),
            onPressed: () => ref.read(isDarkModeProvider.notifier).state = !ref.read(isDarkModeProvider.notifier).state,
          ),
        ],
      ),
      body: PageView(
        controller: _pageController,
        physics: const NeverScrollableScrollPhysics(),
        children: const [
          MarketTab(),
          LiveSimTab(),
          BacktestTab(),
          NewsTab(),
          ChatTab(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() => _currentIndex = index);
          _pageController.jumpToPage(index);
        },
        type: BottomNavigationBarType.fixed,
        backgroundColor: Theme.of(context).cardColor,
        selectedItemColor: Theme.of(context).primaryColor,
        unselectedItemColor: Colors.grey,
        showSelectedLabels: true,
        showUnselectedLabels: true,
        elevation: 0,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.candlestick_chart), label: 'Market'),
          BottomNavigationBarItem(icon: Icon(Icons.smart_toy), label: 'Live Sim'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: 'Backtest'),
          BottomNavigationBarItem(icon: Icon(Icons.newspaper), label: 'News'),
          BottomNavigationBarItem(icon: Icon(Icons.chat_bubble), label: 'Analyst'),
        ],
      ),
    );
  }
}

class PriceBadge extends ConsumerStatefulWidget {
  const PriceBadge({super.key});
  @override
  ConsumerState<PriceBadge> createState() => _PriceBadgeState();
}

class _PriceBadgeState extends ConsumerState<PriceBadge> {
  double? _prevPrice;
  
  @override
  Widget build(BuildContext context) {
    final wsMessage = ref.watch(wsServiceProvider);
    final price = wsMessage?.price ?? 0.0;
    
    if (price == 0) return const SizedBox.shrink();
    
    Color bgColor = Colors.grey.withOpacity(0.2);
    String arrow = '●';
    
    if (_prevPrice != null) {
      if (price > _prevPrice!) {
        bgColor = AppTheme.green;
        arrow = '▲';
      } else if (price < _prevPrice!) {
        bgColor = AppTheme.red;
        arrow = '▼';
      }
    }
    
    _prevPrice = price;
    final formattedPrice = NumberFormat.currency(symbol: '\$', decimalDigits: 2).format(price);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(20),
      ),
      alignment: Alignment.center,
      child: Text(
        'BTC $formattedPrice $arrow',
        style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white),
      ),
    );
  }
}
