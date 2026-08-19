import 'package:flutter/material.dart';

class AppColors {
  // Couleurs principales HiveSense
  static const Color primary      = Color(0xFFFFB300); // Ambre doré
  static const Color primaryDark  = Color(0xFFFF8F00); // Ambre foncé
  static const Color primaryLight = Color(0xFFFFE082); // Ambre clair

  // Couleurs secondaires
  static const Color secondary     = Color(0xFF1B5E20); // Vert forêt
  static const Color secondaryLight = Color(0xFF4CAF50); // Vert clair

  // Fond
  static const Color background    = Color(0xFF121212); // Fond sombre
  static const Color surface       = Color(0xFF1E1E1E); // Surface carte
  static const Color surfaceLight  = Color(0xFF2A2A2A); // Surface claire

  // Statuts
  static const Color success  = Color(0xFF4CAF50);
  static const Color warning  = Color(0xFFFFB300);
  static const Color error    = Color(0xFFEF5350);
  static const Color info     = Color(0xFF42A5F5);

  // Texte
  static const Color textPrimary   = Color(0xFFFFFFFF);
  static const Color textSecondary = Color(0xFFB0B0B0);
  static const Color textHint      = Color(0xFF707070);

  // Statuts des ruches
  static const Color hiveNormal   = Color(0xFF4CAF50);
  static const Color hiveWarning  = Color(0xFFFFB300);
  static const Color hiveCritical = Color(0xFFEF5350);
  static const Color hiveOffline  = Color(0xFF707070);
}

class AppTheme {
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: ColorScheme.dark(
        primary:   AppColors.primary,
        secondary: AppColors.secondary,
        surface:   AppColors.surface,
        error:     AppColors.error,
      ),
      scaffoldBackgroundColor: AppColors.background,

      // AppBar
      appBarTheme: const AppBarTheme(
        backgroundColor:  AppColors.background,
        foregroundColor:  AppColors.textPrimary,
        elevation:        0,
        centerTitle:      false,
        titleTextStyle: TextStyle(
          color:      AppColors.textPrimary,
          fontSize:   20,
          fontWeight: FontWeight.bold,
        ),
      ),

      // Cards
      cardTheme: CardThemeData(
        color:        AppColors.surface,
        elevation:    0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),

      // Boutons
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.black,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(
            horizontal: 24, vertical: 14),
          textStyle: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 15,
          ),
        ),
      ),

      // Bottom Navigation
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor:      AppColors.surface,
        selectedItemColor:    AppColors.primary,
        unselectedItemColor:  AppColors.textHint,
        type:                 BottomNavigationBarType.fixed,
        elevation:            0,
      ),

      // Text
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: AppColors.textPrimary, fontSize: 28,
          fontWeight: FontWeight.bold),
        headlineMedium: TextStyle(
          color: AppColors.textPrimary, fontSize: 22,
          fontWeight: FontWeight.bold),
        headlineSmall: TextStyle(
          color: AppColors.textPrimary, fontSize: 18,
          fontWeight: FontWeight.bold),
        bodyLarge: TextStyle(
          color: AppColors.textPrimary, fontSize: 16),
        bodyMedium: TextStyle(
          color: AppColors.textSecondary, fontSize: 14),
        bodySmall: TextStyle(
          color: AppColors.textHint, fontSize: 12),
        labelLarge: TextStyle(
          color: AppColors.primary, fontSize: 14,
          fontWeight: FontWeight.bold),
      ),

      // Divider
      dividerTheme: const DividerThemeData(
        color:     AppColors.surfaceLight,
        thickness: 1,
      ),

      // Input
      inputDecorationTheme: InputDecorationTheme(
        filled:      true,
        fillColor:   AppColors.surfaceLight,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        hintStyle: const TextStyle(color: AppColors.textHint),
      ),
    );
  }
}