class ChatMessage {
  final String role; // "user" | "assistant" | "system"
  final String content;
  final DateTime timestamp;
  final bool isError;
  final List<String> suggestedQuestions;
  final String? dataConfidence;
  final String? currentVerdict;
  final int? currentScore;

  ChatMessage({
    required this.role,
    required this.content,
    required this.timestamp,
    this.isError = false,
    this.suggestedQuestions = const [],
    this.dataConfidence,
    this.currentVerdict,
    this.currentScore,
  });

  factory ChatMessage.fromJson(Map<String, dynamic> json, String role) {
    return ChatMessage(
      role: role,
      content: json['answer'] ?? json['content'] ?? '',
      timestamp: DateTime.now(),
      suggestedQuestions: List<String>.from(json['suggested_questions'] ?? []),
      dataConfidence: json['data_confidence']?.toString(),
      currentVerdict: json['current_verdict']?.toString(),
      currentScore: json['current_score'] as int?,
    );
  }
}