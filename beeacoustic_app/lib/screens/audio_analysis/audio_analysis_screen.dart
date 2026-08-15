import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../../core/theme/app_theme.dart';
import '../../core/constants/app_constants.dart';

class AudioAnalysisScreen extends StatefulWidget {
  const AudioAnalysisScreen({super.key});
  @override
  State<AudioAnalysisScreen> createState() => _AudioAnalysisScreenState();
}

class _AudioAnalysisScreenState extends State<AudioAnalysisScreen> {
  String?  _fileName;
  bool     _isLoading   = false;
  bool     _hasResult   = false;
  String?  _errorMsg;
  Map<String, dynamic>? _result;

  // ── Sélection et envoi du fichier ─────────────────────────────────────────
  Future<void> _pickAndAnalyze() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav', 'mp3', 'ogg'],
      withData: true,
    );

    if (picked == null || picked.files.isEmpty) return;
    final file = picked.files.first;

    setState(() {
      _fileName  = file.name;
      _isLoading = true;
      _hasResult = false;
      _errorMsg  = null;
      _result    = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${AppConstants.model1BaseUrl}/predict'),
      );
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        file.bytes!,
        filename: file.name,
      ));

      final response = await request.send().timeout(
        Duration(seconds: AppConstants.apiTimeout));
      final body = await response.stream.bytesToString();
      final json = jsonDecode(body);

      setState(() {
        _result    = json;
        _hasResult = true;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMsg  = 'Impossible de contacter l\'API.\nVérifiez que le serveur est démarré.';
        _isLoading = false;
      });
    }
  }

  void _reset() {
    setState(() {
      _fileName  = null;
      _hasResult = false;
      _errorMsg  = null;
      _result    = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              const SizedBox(height: 24),
              _buildUploadSection(),
              const SizedBox(height: 24),
              if (_isLoading)   _buildLoading(),
              if (_errorMsg != null) _buildError(),
              if (_hasResult && _result != null) _buildResults(),
              const SizedBox(height: 80),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Analyse Audio',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          )),
        const SizedBox(height: 4),
        const Text(
          'Uploadez un enregistrement audio de votre ruche '
          'pour obtenir une analyse par intelligence artificielle.',
          style: TextStyle(
            color: AppColors.textSecondary,
            fontSize: 13,
          )),
        const SizedBox(height: 16),
        // Info modèles
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
                color: AppColors.primary.withOpacity(0.3)),
          ),
          child: Row(
            children: [
              const Icon(Icons.info_rounded,
                  color: AppColors.primary, size: 18),
              const SizedBox(width: 10),
              Expanded(
                child: RichText(
                  text: const TextSpan(
                    style: TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 12,
                    ),
                    children: [
                      TextSpan(text: 'Modèle 1 : ',
                        style: TextStyle(
                          color: AppColors.primary,
                          fontWeight: FontWeight.bold,
                        )),
                      TextSpan(
                        text: 'Détection état de la reine (4 classes) — ',
                      ),
                      TextSpan(text: 'Accuracy 91.5%',
                        style: TextStyle(
                          color: AppColors.success,
                          fontWeight: FontWeight.bold,
                        )),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ── Zone upload ───────────────────────────────────────────────────────────
  Widget _buildUploadSection() {
    return GestureDetector(
      onTap: _isLoading ? null : _pickAndAnalyze,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
        decoration: BoxDecoration(
          color: _fileName != null
              ? AppColors.primary.withOpacity(0.08)
              : AppColors.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: _fileName != null
                ? AppColors.primary.withOpacity(0.5)
                : AppColors.surfaceLight,
            width: 2,
            strokeAlign: BorderSide.strokeAlignInside,
          ),
        ),
        child: Column(
          children: [
            Container(
              width: 64, height: 64,
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(
                _fileName != null
                    ? Icons.audio_file_rounded
                    : Icons.upload_file_rounded,
                color: AppColors.primary,
                size: 32,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              _fileName ?? 'Appuyez pour sélectionner un fichier audio',
              style: TextStyle(
                color: _fileName != null
                    ? AppColors.textPrimary
                    : AppColors.textSecondary,
                fontSize: 14,
                fontWeight: _fileName != null
                    ? FontWeight.bold
                    : FontWeight.normal,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 6),
            Text(
              _fileName != null
                  ? 'Tapez pour changer de fichier'
                  : 'Formats acceptés : WAV, MP3, OGG',
              style: const TextStyle(
                color: AppColors.textHint,
                fontSize: 11,
              ),
            ),
            if (_fileName != null && !_isLoading) ...[
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: _pickAndAnalyze,
                    icon: const Icon(Icons.psychology_rounded, size: 18),
                    label: const Text('Analyser'),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton.icon(
                    onPressed: _reset,
                    icon: const Icon(Icons.refresh_rounded,
                        size: 18, color: AppColors.textSecondary),
                    label: const Text('Réinitialiser',
                      style: TextStyle(color: AppColors.textSecondary)),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(
                          color: AppColors.surfaceLight),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ── Loading ───────────────────────────────────────────────────────────────
  Widget _buildLoading() {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Column(
        children: [
          CircularProgressIndicator(color: AppColors.primary),
          SizedBox(height: 16),
          Text('Analyse en cours...',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 15,
              fontWeight: FontWeight.bold,
            )),
          SizedBox(height: 4),
          Text('Le modèle analyse le signal audio',
            style: TextStyle(
              color: AppColors.textSecondary,
              fontSize: 12,
            )),
        ],
      ),
    );
  }

  // ── Erreur ────────────────────────────────────────────────────────────────
  Widget _buildError() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.error.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
            color: AppColors.error.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_rounded,
              color: AppColors.error, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_errorMsg!,
              style: const TextStyle(
                color: AppColors.textSecondary,
                fontSize: 13,
              )),
          ),
        ],
      ),
    );
  }

  // ── Résultats ─────────────────────────────────────────────────────────────
  Widget _buildResults() {
    final prediction   = _result!['prediction'] as String;
    final confidence   = (_result!['confidence'] as num).toDouble();
    final recommendation = _result!['recommendation'] as String;
    final probs = Map<String, double>.from(
      (_result!['probabilities'] as Map).map(
        (k, v) => MapEntry(k as String, (v as num).toDouble())));
    final windows = _result!['windows_count'] as int;

    final color = _predictionColor(prediction);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Résultat principal
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: color.withOpacity(0.4)),
          ),
          child: Column(
            children: [
              Container(
                width: 64, height: 64,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  shape: BoxShape.circle,
                ),
                child: Icon(_predictionIcon(prediction),
                    color: color, size: 34),
              ),
              const SizedBox(height: 12),
              Text(prediction,
                style: TextStyle(
                  color: color,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 6),
              Text('Confiance : ${(confidence * 100).toStringAsFixed(1)}%',
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 14,
                )),
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(
                  value: confidence,
                  backgroundColor: AppColors.surfaceLight,
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                  minHeight: 8,
                ),
              ),
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.recommend_rounded,
                        color: AppColors.primary, size: 16),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(recommendation,
                        style: const TextStyle(
                          color: AppColors.textSecondary,
                          fontSize: 12,
                        )),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              Text('Analysé sur $windows fenêtres de 5 secondes',
                style: const TextStyle(
                  color: AppColors.textHint,
                  fontSize: 11,
                )),
            ],
          ),
        ),
        const SizedBox(height: 16),
        // Distribution probabilités
        const Text('Distribution des probabilités',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 15,
            fontWeight: FontWeight.bold,
          )),
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            children: probs.entries.map((e) {
              final isMax = e.value == probs.values
                  .reduce((a, b) => a > b ? a : b);
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment:
                          MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Text(e.key,
                            style: TextStyle(
                              color: isMax
                                  ? AppColors.textPrimary
                                  : AppColors.textSecondary,
                              fontSize: 12,
                              fontWeight: isMax
                                  ? FontWeight.bold
                                  : FontWeight.normal,
                            )),
                        ),
                        Text(
                          '${(e.value * 100).toStringAsFixed(1)}%',
                          style: TextStyle(
                            color: isMax
                                ? AppColors.primary
                                : AppColors.textHint,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          )),
                      ],
                    ),
                    const SizedBox(height: 4),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: e.value,
                        backgroundColor: AppColors.surfaceLight,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          isMax
                              ? AppColors.primary
                              : AppColors.textHint,
                        ),
                        minHeight: 5,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
        const SizedBox(height: 16),
        // Nouvelle analyse
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: _reset,
            icon: const Icon(Icons.refresh_rounded,
                color: AppColors.primary),
            label: const Text('Nouvelle analyse',
              style: TextStyle(color: AppColors.primary)),
            style: OutlinedButton.styleFrom(
              side: const BorderSide(color: AppColors.primary),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
              padding: const EdgeInsets.symmetric(vertical: 14),
            ),
          ),
        ),
      ],
    );
  }

  Color _predictionColor(String prediction) {
    if (prediction.contains('Not Present') ||
        prediction.contains('absente'))    return AppColors.error;
    if (prediction.contains('Rejected') ||
        prediction.contains('rejetée'))   return AppColors.warning;
    if (prediction.contains('Accepted') ||
        prediction.contains('acceptée'))  return AppColors.info;
    return AppColors.success;
  }

  IconData _predictionIcon(String prediction) {
    if (prediction.contains('Not Present') ||
        prediction.contains('absente'))   return Icons.error_rounded;
    if (prediction.contains('Rejected'))  return Icons.warning_rounded;
    if (prediction.contains('Accepted'))  return Icons.new_releases_rounded;
    return Icons.check_circle_rounded;
  }
}