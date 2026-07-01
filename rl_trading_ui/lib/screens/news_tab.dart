import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';
import '../models/news_model.dart';
import '../app_theme.dart';
import '../providers.dart'; // Ensure this path points correctly to your providers.dart file

class NewsTab extends ConsumerWidget {
  const NewsTab({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final newsAsyncValue = ref.watch(newsProvider);

    return RefreshIndicator(
      onRefresh: () => ref.refresh(newsProvider.future),
      child: newsAsyncValue.when(
        data: (newsList) => Column(
          children: [
            _buildSentimentSummaryBar(newsList),
            _buildSourceFilterChips(),
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: newsList.length,
                itemBuilder: (context, index) => _buildNewsCard(context, newsList[index]),
              ),
            ),
          ],
        ),
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stack) => Center(child: Text('Error: $error')),
      ),
    );
  }

  Widget _buildSentimentSummaryBar(List<NewsModel> news) {
    int positive = 0;
    for (var n in news) {
      if (n.sentiment == 'positive') positive++;
    }
    double positiveRatio = news.isEmpty ? 0 : positive / news.length;
    int percent = (positiveRatio * 100).round();
    String marketSentiment = positiveRatio >= 0.5 ? 'BULLISH' : 'BEARISH';
    Color barColor = positiveRatio >= 0.5 ? AppTheme.green : AppTheme.red;

    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: barColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: barColor.withOpacity(0.2)),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Market Sentiment: $marketSentiment', style: TextStyle(fontWeight: FontWeight.bold, color: barColor)),
              Text('$percent% positive', style: TextStyle(fontSize: 12, color: barColor)),
            ],
          ),
          const SizedBox(height: 8),
          LinearProgressIndicator(
            value: positiveRatio,
            backgroundColor: Colors.white10,
            valueColor: AlwaysStoppedAnimation<Color>(barColor),
          ),
        ],
      ),
    );
  }

  Widget _buildSourceFilterChips() {
    final sources = ['All', 'CoinDesk', 'CoinTelegraph', 'Bloomberg', 'Reuters'];
    return SizedBox(
      height: 50,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemCount: sources.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) => ChoiceChip(
          label: Text(sources[index]),
          selected: index == 0,
          onSelected: (_) {},
        ),
      ),
    );
  }

  Widget _buildNewsCard(BuildContext context, NewsModel news) {
    final timeDiff = DateTime.now().difference(news.publishedAt);
    String timeAgo;
    if (timeDiff.inMinutes < 60) {
      timeAgo = '${timeDiff.inMinutes}m ago';
    } else {
      timeAgo = '${timeDiff.inHours}h ago';
    }

    Color sentimentColor = AppTheme.green;
    if (news.sentiment == 'negative') sentimentColor = AppTheme.red;
    if (news.sentiment == 'neutral') sentimentColor = Colors.grey;

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      color: Theme.of(context).cardColor,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(news.source.toUpperCase(), style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: AppTheme.primary)),
                Text(timeAgo, style: const TextStyle(fontSize: 10, color: Colors.grey)),
              ],
            ),
            const SizedBox(height: 8),
            Text(news.title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.circle, color: sentimentColor, size: 8),
                const SizedBox(width: 4),
                Text(news.sentiment.toUpperCase(), style: TextStyle(fontSize: 10, color: sentimentColor)),
                const Spacer(),
                TextButton(
                  onPressed: () async {
                    final uri = Uri.parse(news.url);
                    if (await canLaunchUrl(uri)) {
                      await launchUrl(uri);
                    }
                  },
                  child: const Text('Read More')
                )
              ],
            )
          ],
        ),
      ),
    );
  }
}
