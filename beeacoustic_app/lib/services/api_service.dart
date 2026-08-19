import 'dart:convert';
import 'package:http/http.dart' as http;
import '../core/constants/app_constants.dart';
import '../models/hive_model.dart';

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  // ── Health Check ──────────────────────────────────────────────────────────
  Future<Map<String, dynamic>> checkModel1Health() async {
    try {
      final resp = await http
          .get(Uri.parse('${AppConstants.model1BaseUrl}/health'))
          .timeout(Duration(seconds: AppConstants.apiTimeout));
      if (resp.statusCode == 200) {
        return {'ok': true, ...jsonDecode(resp.body)};
      }
      return {'ok': false, 'error': 'Statut ${resp.statusCode}'};
    } catch (e) {
      return {'ok': false, 'error': 'Non joignable'};
    }
  }

  Future<Map<String, dynamic>> checkModel2Health() async {
    try {
      final resp = await http
          .get(Uri.parse('${AppConstants.model2BaseUrl}/health'))
          .timeout(Duration(seconds: AppConstants.apiTimeout));
      if (resp.statusCode == 200) {
        return {'ok': true, ...jsonDecode(resp.body)};
      }
      return {'ok': false, 'error': 'Statut ${resp.statusCode}'};
    } catch (e) {
      return {'ok': false, 'error': 'Non joignable'};
    }
  }

  // ── Modèle 1 — Prédiction état de la reine ────────────────────────────────
  Future<AIPrediction?> predictQueenStatus(
      List<int> fileBytes, String fileName) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${AppConstants.model1BaseUrl}/predict'),
      );
      request.files.add(http.MultipartFile.fromBytes(
        'file', fileBytes, filename: fileName));

      final response = await request.send()
          .timeout(Duration(seconds: AppConstants.apiTimeout));
      final body = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final json = jsonDecode(body);
        return _parseQueenPrediction(json);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  // ── Modèle 2 — Détection anomalie acoustique ──────────────────────────────
  Future<AIPrediction?> predictAnomaly(
      List<int> fileBytes, String fileName) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${AppConstants.model2BaseUrl}/predict'),
      );
      request.files.add(http.MultipartFile.fromBytes(
        'file', fileBytes, filename: fileName));

      final response = await request.send()
          .timeout(Duration(seconds: AppConstants.apiTimeout));
      final body = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final json = jsonDecode(body);
        return _parseAnomalyPrediction(json);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  // ── Prédiction batch (plusieurs fichiers) ─────────────────────────────────
  Future<List<AIPrediction>> predictBatch(
      List<Map<String, dynamic>> files) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${AppConstants.model1BaseUrl}/predict/batch'),
      );
      for (final file in files) {
        request.files.add(http.MultipartFile.fromBytes(
          'files',
          file['bytes'] as List<int>,
          filename: file['name'] as String,
        ));
      }
      final response = await request.send()
          .timeout(Duration(seconds: AppConstants.apiTimeout));
      final body = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final json    = jsonDecode(body);
        final results = json['results'] as List;
        return results
            .map((r) => _parseQueenPrediction(r))
            .whereType<AIPrediction>()
            .toList();
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  // ── Parsers ───────────────────────────────────────────────────────────────
  AIPrediction? _parseQueenPrediction(Map<String, dynamic> json) {
    try {
      final labelMap = {
        'Queen Not Present':                AppConstants.queenStatus[0]!,
        'Queen Present - Newly Accepted':   AppConstants.queenStatus[1]!,
        'Queen Present - Rejected':         AppConstants.queenStatus[2]!,
        'Queen Present - Original Queen':   AppConstants.queenStatus[3]!,
      };

      final rawClass = json['prediction'] as String? ?? '';
      final label    = json['label'] as int? ?? 0;
      final className = labelMap[rawClass] ??
          AppConstants.queenStatus[label] ?? rawClass;

      final rawProbs = json['probabilities'] as Map<String, dynamic>? ?? {};
      final probs    = rawProbs.map((k, v) =>
          MapEntry(labelMap[k] ?? k, (v as num).toDouble()));

      return AIPrediction(
        label:          label,
        className:      className,
        confidence:     (json['confidence'] as num?)?.toDouble() ?? 0,
        recommendation: json['recommendation'] as String? ?? '',
        probabilities:  probs,
      );
    } catch (_) {
      return null;
    }
  }

  AIPrediction? _parseAnomalyPrediction(Map<String, dynamic> json) {
    try {
      final label     = json['label'] as int? ?? 0;
      final className = AppConstants.anomalyStatus[label] ??
          json['class'] as String? ?? '';

      return AIPrediction(
        label:          label,
        className:      className,
        confidence:     (json['confidence'] as num?)?.toDouble() ?? 0,
        recommendation: json['recommendation'] as String? ?? '',
        probabilities:  {},
      );
    } catch (_) {
      return null;
    }
  }
}