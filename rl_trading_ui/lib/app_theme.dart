import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color darkBg = Color(0xFF0A0A0F);
  static const Color darkCard = Color(0xFF13131A);
  static const Color darkBorder = Color(0xFF2A2A3A);
  
  static const Color lightBg = Color(0xFFF5F5FA);
  static const Color lightCard = Color(0xFFFFFFFF);
  static const Color lightBorder = Color(0xFFE0E0EE);

  static const Color primary = Color(0xFF6C63FF);
  static const Color green = Color(0xFF00D4A4);
  static const Color red = Color(0xFFFF4757);
  static const Color amber = Color(0xFFFFB800);

  static ThemeData darkTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: darkBg,
    cardColor: darkCard,
    primaryColor: primary,
    colorScheme: const ColorScheme.dark(
      primary: primary,
      secondary: green,
      error: red,
      background: darkBg,
      surface: darkCard,
    ),
    dividerColor: darkBorder,
    textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme),
  );

  static ThemeData lightTheme = ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    scaffoldBackgroundColor: lightBg,
    cardColor: lightCard,
    primaryColor: primary,
    colorScheme: const ColorScheme.light(
      primary: primary,
      secondary: green,
      error: red,
      background: lightBg,
      surface: lightCard,
    ),
    dividerColor: lightBorder,
    textTheme: GoogleFonts.interTextTheme(ThemeData.light().textTheme),
  );

  static Color verdictColor(String verdict) {
    switch (verdict.toUpperCase()) {
      case 'STRONG BUY':
        return green;
      case 'BUY':
        return const Color(0xFF00AA88);
      case 'NEUTRAL':
        return amber;
      case 'WAIT':
        return const Color(0xFFFF8C00);
      case 'EXIT':
        return red;
      default:
        return Colors.grey;
    }
  }
}
