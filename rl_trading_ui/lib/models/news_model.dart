class NewsModel {
  final String title;
  final String source;
  final String url;
  final DateTime publishedAt;
  final String sentiment; // "positive" | "negative" | "neutral"
  final String summary;

  NewsModel({
    required this.title,
    required this.source,
    required this.url,
    required this.publishedAt,
    required this.sentiment,
    required this.summary,
  });

  factory NewsModel.fromJson(Map<String, dynamic> json) {
    return NewsModel(
      title: json['title'] ?? '',
      source: json['source'] ?? '',
      url: json['url'] ?? '',
      publishedAt: json['published_at'] != null ? DateTime.parse(json['published_at']) : DateTime.now(),
      sentiment: json['sentiment'] ?? 'neutral',
      summary: json['summary'] ?? '',
    );
  }
}
