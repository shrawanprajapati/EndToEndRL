import 'package:flutter/material.dart';
import '../app_theme.dart';

class VerdictBadge extends StatelessWidget {
  final String verdict;

  const VerdictBadge({super.key, required this.verdict});

  @override
  Widget build(BuildContext context) {
    final color = AppTheme.verdictColor(verdict);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(
        verdict.toUpperCase(),
        style: TextStyle(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
