import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../core/constants/app_constants.dart';

class HiveModel {
  final String id;
  final String name;
  final String location;
  final double latitude;
  final double longitude;
  final DateTime lastUpdate;
  final SensorData sensors;
  final AIPrediction? queenPrediction;
  final AIPrediction? anomalyPrediction;
  final List<AlertModel> recentAlerts;
  final bool isOnline;

  HiveModel({
    required this.id,
    required this.name,
    required this.location,
    required this.latitude,
    required this.longitude,
    required this.lastUpdate,
    required this.sensors,
    this.queenPrediction,
    this.anomalyPrediction,
    this.recentAlerts = const [],
    this.isOnline = true,
  });

  String get status {
    if (!isOnline) return AppConstants.statusOffline;
    if (queenPrediction?.label == 0) return AppConstants.statusCritical;
    if (queenPrediction?.label == 2) return AppConstants.statusWarning;
    if (sensors.temperature > AppConstants.tempMaxAlert ||
        sensors.temperature < AppConstants.tempMinAlert)
      return AppConstants.statusWarning;
    return AppConstants.statusNormal;
  }

  Color get statusColor {
    switch (status) {
      case 'Normal':     return AppColors.hiveNormal;
      case 'Attention':  return AppColors.hiveWarning;
      case 'Critique':   return AppColors.hiveCritical;
      default:           return AppColors.hiveOffline;
    }
  }

  IconData get statusIcon {
    switch (status) {
      case 'Normal':     return Icons.check_circle;
      case 'Attention':  return Icons.warning_rounded;
      case 'Critique':   return Icons.error_rounded;
      default:           return Icons.wifi_off;
    }
  }

  // Données simulées pour le développement
  static List<HiveModel> getMockHives() {
    return [
      HiveModel(
        id: 'hive_001',
        name: 'Ruche Alpha',
        location: 'Secteur Nord',
        latitude: 36.8065,
        longitude: 10.1815,
        lastUpdate: DateTime.now().subtract(const Duration(minutes: 5)),
        isOnline: true,
        sensors: SensorData(
          temperature:       34.2,
          humidity:          62.5,
          pressure:          1013.0,
          weatherTemp:       28.0,
          weatherHumidity:   55.0,
          windSpeed:         3.5,
          batteryLevel:      85,
          signalStrength:    -65,
        ),
        queenPrediction: AIPrediction(
          label:       3,
          className:   'Reine présente',
          confidence:  0.952,
          recommendation: 'État normal. Aucune action requise.',
          probabilities: {
            'Reine absente':           0.010,
            'Nouvelle reine acceptée': 0.020,
            'Reine rejetée':           0.018,
            'Reine présente':          0.952,
          },
        ),
        anomalyPrediction: AIPrediction(
          label:       0,
          className:   'Activité normale',
          confidence:  0.91,
          recommendation: 'La colonie est active et saine.',
          probabilities: {
            'Activité normale':  0.91,
            'Absence d\'abeilles': 0.04,
            'Reine manquante':   0.05,
          },
        ),
        recentAlerts: [],
      ),
      HiveModel(
        id: 'hive_002',
        name: 'Ruche Beta',
        location: 'Secteur Est',
        latitude: 36.8100,
        longitude: 10.1850,
        lastUpdate: DateTime.now().subtract(const Duration(minutes: 12)),
        isOnline: true,
        sensors: SensorData(
          temperature:       35.8,
          humidity:          71.0,
          pressure:          1012.0,
          weatherTemp:       29.0,
          weatherHumidity:   58.0,
          windSpeed:         5.2,
          batteryLevel:      62,
          signalStrength:    -72,
        ),
        queenPrediction: AIPrediction(
          label:       0,
          className:   'Reine absente',
          confidence:  0.881,
          recommendation: 'URGENT : La reine est absente. Inspectez la ruche immédiatement.',
          probabilities: {
            'Reine absente':           0.881,
            'Nouvelle reine acceptée': 0.072,
            'Reine rejetée':           0.031,
            'Reine présente':          0.016,
          },
        ),
        recentAlerts: [
          AlertModel(
            id: 'alert_001',
            hiveId: 'hive_002',
            hiveName: 'Ruche Beta',
            type: AlertType.queenMissing,
            message: 'Reine absente détectée',
            timestamp: DateTime.now().subtract(const Duration(hours: 1)),
            isRead: false,
          ),
        ],
      ),
      HiveModel(
        id: 'hive_003',
        name: 'Ruche Gamma',
        location: 'Secteur Ouest',
        latitude: 36.8030,
        longitude: 10.1780,
        lastUpdate: DateTime.now().subtract(const Duration(hours: 2)),
        isOnline: false,
        sensors: SensorData(
          temperature:       33.0,
          humidity:          65.0,
          pressure:          1011.0,
          weatherTemp:       27.0,
          weatherHumidity:   52.0,
          windSpeed:         2.1,
          batteryLevel:      12,
          signalStrength:    -95,
        ),
        recentAlerts: [
          AlertModel(
            id: 'alert_002',
            hiveId: 'hive_003',
            hiveName: 'Ruche Gamma',
            type: AlertType.offline,
            message: 'Device hors ligne — batterie critique (12%)',
            timestamp: DateTime.now().subtract(const Duration(hours: 2)),
            isRead: false,
          ),
        ],
      ),
    ];
  }
}

class SensorData {
  final double temperature;
  final double humidity;
  final double pressure;
  final double weatherTemp;
  final double weatherHumidity;
  final double windSpeed;
  final int    batteryLevel;
  final int    signalStrength;

  SensorData({
    required this.temperature,
    required this.humidity,
    required this.pressure,
    required this.weatherTemp,
    required this.weatherHumidity,
    required this.windSpeed,
    required this.batteryLevel,
    required this.signalStrength,
  });
}

class AIPrediction {
  final int    label;
  final String className;
  final double confidence;
  final String recommendation;
  final Map<String, double> probabilities;

  AIPrediction({
    required this.label,
    required this.className,
    required this.confidence,
    required this.recommendation,
    required this.probabilities,
  });
}

class AlertModel {
  final String    id;
  final String    hiveId;
  final String    hiveName;
  final AlertType type;
  final String    message;
  final DateTime  timestamp;
  bool            isRead;

  AlertModel({
    required this.id,
    required this.hiveId,
    required this.hiveName,
    required this.type,
    required this.message,
    required this.timestamp,
    this.isRead = false,
  });
}

enum AlertType {
  queenMissing,
  queenRejected,
  temperatureHigh,
  temperatureLow,
  humidityHigh,
  anomaly,
  offline,
  batteryLow,
}