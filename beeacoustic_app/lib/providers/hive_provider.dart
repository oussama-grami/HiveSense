import 'package:flutter/material.dart';
import '../models/hive_model.dart';
import '../services/api_service.dart';

class HiveProvider extends ChangeNotifier {
  final ApiService _api = ApiService();

  // ── État ──────────────────────────────────────────────────────────────────
  List<HiveModel> _hives        = [];
  bool            _isLoading    = false;
  bool            _model1Online = false;
  bool            _model2Online = false;
  String?         _error;

  // ── Getters ───────────────────────────────────────────────────────────────
  List<HiveModel> get hives        => _hives;
  bool            get isLoading    => _isLoading;
  bool            get model1Online => _model1Online;
  bool            get model2Online => _model2Online;
  String?         get error        => _error;

  int get totalHives    => _hives.length;
  int get normalHives   => _hives.where((h) => h.status == 'Normal').length;
  int get warningHives  => _hives.where((h) => h.status == 'Attention').length;
  int get criticalHives => _hives.where((h) => h.status == 'Critique').length;
  int get offlineHives  => _hives.where((h) => h.status == 'Hors ligne').length;

  List<AlertModel> get allAlerts => _hives
      .expand((h) => h.recentAlerts)
      .toList()
    ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

  List<AlertModel> get unreadAlerts =>
      allAlerts.where((a) => !a.isRead).toList();

  int get unreadCount => unreadAlerts.length;

  HiveModel? getHiveById(String id) {
    try {
      return _hives.firstWhere((h) => h.id == id);
    } catch (_) {
      return null;
    }
  }

  // ── Initialisation ────────────────────────────────────────────────────────
  Future<void> initialize() async {
    _isLoading = true;
    notifyListeners();

    // Charger les ruches (mock pour l'instant)
    await Future.delayed(const Duration(milliseconds: 500));
    _hives = HiveModel.getMockHives();

    // Vérifier les APIs en parallèle
    await checkApiStatus();

    _isLoading = false;
    notifyListeners();
  }

  // ── Rafraîchir ────────────────────────────────────────────────────────────
  Future<void> refresh() async {
    _isLoading = true;
    notifyListeners();

    await Future.delayed(const Duration(seconds: 1));
    _hives = HiveModel.getMockHives();

    _isLoading = false;
    notifyListeners();
  }

  // ── Vérifier statut APIs ──────────────────────────────────────────────────
  Future<void> checkApiStatus() async {
  try {
    final result = await _api.checkModel1Health();
    _model1Online = result['ok'] as bool? ?? false;
  } catch (_) {
    _model1Online = false;
  }

  try {
    final result = await _api.checkModel2Health();
    _model2Online = result['ok'] as bool? ?? false;
  } catch (_) {
    _model2Online = false;
  }

  notifyListeners();
}

  // ── Prédiction audio ──────────────────────────────────────────────────────
  Future<AIPrediction?> analyzeAudio(
      List<int> bytes, String fileName) async {
    return await _api.predictQueenStatus(bytes, fileName);
  }

  Future<AIPrediction?> analyzeAnomaly(
      List<int> bytes, String fileName) async {
    return await _api.predictAnomaly(bytes, fileName);
  }

  // ── Gestion des alertes ───────────────────────────────────────────────────
  void markAlertRead(String alertId) {
    for (final hive in _hives) {
      for (final alert in hive.recentAlerts) {
        if (alert.id == alertId) {
          alert.isRead = true;
          notifyListeners();
          return;
        }
      }
    }
  }

  void markAllAlertsRead() {
    for (final hive in _hives) {
      for (final alert in hive.recentAlerts) {
        alert.isRead = true;
      }
    }
    notifyListeners();
  }

  // ── Mise à jour prédiction IA d'une ruche ─────────────────────────────────
  void updateHivePrediction(String hiveId, AIPrediction prediction,
      {bool isAnomaly = false}) {
    final index = _hives.indexWhere((h) => h.id == hiveId);
    if (index == -1) return;

    final hive = _hives[index];
    _hives[index] = HiveModel(
      id:                hive.id,
      name:              hive.name,
      location:          hive.location,
      latitude:          hive.latitude,
      longitude:         hive.longitude,
      lastUpdate:        DateTime.now(),
      sensors:           hive.sensors,
      queenPrediction:   isAnomaly ? hive.queenPrediction : prediction,
      anomalyPrediction: isAnomaly ? prediction : hive.anomalyPrediction,
      recentAlerts:      hive.recentAlerts,
      isOnline:          hive.isOnline,
    );
    notifyListeners();
  }
}