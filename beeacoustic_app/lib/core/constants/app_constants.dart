class AppConstants {
  // Nom de l'application
  static const String appName    = 'HiveSense';
  static const String appVersion = '1.0.0';
  static const String appTagline = 'Surveillance intelligente de vos ruches';

  // URLs des APIs (à adapter selon l'IP de ton serveur)
  static const String model1BaseUrl = 'http://192.168.1.6:5000';
  static const String model2BaseUrl = 'http://192.168.1.6:5001';

  // Timeouts
  static const int apiTimeout = 30; // secondes

  // Seuils d'alerte température
  static const double tempMinAlert  = 32.0;
  static const double tempMaxAlert  = 36.0;
  static const double tempOptimal   = 34.5;

  // Seuils d'alerte humidité
  static const double humidityMin   = 50.0;
  static const double humidityMax   = 80.0;

  // Seuils confiance modèle
  static const double confidenceHigh   = 0.85;
  static const double confidenceMedium = 0.60;

  // Statuts des ruches
  static const String statusNormal   = 'Normal';
  static const String statusWarning  = 'Attention';
  static const String statusCritical = 'Critique';
  static const String statusOffline  = 'Hors ligne';

  // Classes Modèle 1
  static const Map<int, String> queenStatus = {
    0: 'Reine absente',
    1: 'Nouvelle reine acceptée',
    2: 'Reine rejetée',
    3: 'Reine présente',
  };

  // Classes Modèle 2
  static const Map<int, String> anomalyStatus = {
    0: 'Activité normale',
    1: 'Absence d\'abeilles',
    2: 'Reine manquante',
  };
}