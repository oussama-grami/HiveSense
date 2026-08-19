import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';
import '../../core/constants/app_constants.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  // Paramètres
  bool _notificationsEnabled = true;
  bool _soundEnabled         = true;
  bool _autoRefresh          = true;
  int  _refreshInterval      = 5;
  double _tempAlertMin       = 32.0;
  double _tempAlertMax       = 36.0;
  double _humidityAlertMin   = 50.0;
  double _humidityAlertMax   = 80.0;
  double _confidenceThreshold = 0.75;
  String _model1Url          = AppConstants.model1BaseUrl;
  String _model2Url          = AppConstants.model2BaseUrl;

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
              _buildSection('Notifications', Icons.notifications_rounded, [
                _switchTile('Notifications push',
                    'Recevoir des alertes en temps réel',
                    _notificationsEnabled,
                    (v) => setState(() => _notificationsEnabled = v)),
                _switchTile('Son',
                    'Activer le son des alertes',
                    _soundEnabled,
                    (v) => setState(() => _soundEnabled = v)),
              ]),
              const SizedBox(height: 16),
              _buildSection('Actualisation', Icons.refresh_rounded, [
                _switchTile('Actualisation auto',
                    'Mettre à jour les données automatiquement',
                    _autoRefresh,
                    (v) => setState(() => _autoRefresh = v)),
                _sliderTile(
                  'Intervalle d\'actualisation',
                  '$_refreshInterval min',
                  _refreshInterval.toDouble(),
                  1, 30,
                  (v) => setState(() => _refreshInterval = v.toInt()),
                ),
              ]),
              const SizedBox(height: 16),
              _buildSection('Seuils d\'alerte température',
                  Icons.thermostat_rounded, [
                _sliderTile(
                  'Température minimale',
                  '${_tempAlertMin.toStringAsFixed(1)}°C',
                  _tempAlertMin, 28, 34,
                  (v) => setState(() => _tempAlertMin = v),
                ),
                _sliderTile(
                  'Température maximale',
                  '${_tempAlertMax.toStringAsFixed(1)}°C',
                  _tempAlertMax, 34, 40,
                  (v) => setState(() => _tempAlertMax = v),
                ),
              ]),
              const SizedBox(height: 16),
              _buildSection('Seuils d\'alerte humidité',
                  Icons.water_drop_rounded, [
                _sliderTile(
                  'Humidité minimale',
                  '${_humidityAlertMin.toStringAsFixed(0)}%',
                  _humidityAlertMin, 30, 60,
                  (v) => setState(() => _humidityAlertMin = v),
                ),
                _sliderTile(
                  'Humidité maximale',
                  '${_humidityAlertMax.toStringAsFixed(0)}%',
                  _humidityAlertMax, 70, 95,
                  (v) => setState(() => _humidityAlertMax = v),
                ),
              ]),
              const SizedBox(height: 16),
              _buildSection('Intelligence artificielle',
                  Icons.psychology_rounded, [
                _sliderTile(
                  'Seuil de confiance minimum',
                  '${(_confidenceThreshold * 100).toStringAsFixed(0)}%',
                  _confidenceThreshold, 0.5, 0.99,
                  (v) => setState(() => _confidenceThreshold = v),
                ),
                _infoTile('Modèle 1 — État de la reine',
                    'Accuracy : 91.5% | F1 : 0.915 | AUC : 0.989',
                    Icons.check_circle_rounded, AppColors.success),
                _infoTile('Modèle 2 — Anomalie acoustique',
                    'En cours de développement',
                    Icons.hourglass_empty_rounded, AppColors.warning),
              ]),
              const SizedBox(height: 16),
              _buildSection('Configuration API',
                  Icons.cloud_rounded, [
                _urlTile('URL Modèle 1', _model1Url,
                    (v) => setState(() => _model1Url = v)),
                _urlTile('URL Modèle 2', _model2Url,
                    (v) => setState(() => _model2Url = v)),
                _actionTile(
                  'Tester la connexion',
                  'Vérifier que les APIs sont accessibles',
                  Icons.wifi_tethering_rounded,
                  AppColors.info,
                  _testConnection,
                ),
              ]),
              const SizedBox(height: 16),
              _buildSection('À propos', Icons.info_rounded, [
                _infoTile('HiveSense',
                    'Version ${AppConstants.appVersion}',
                    Icons.hexagon_rounded, AppColors.primary),
                _infoTile('Modèles IA',
                    'Développés sur dataset SBCM (Kaggle)',
                    Icons.dataset_rounded, AppColors.textSecondary),
                _infoTile('Framework',
                    'Flutter 3.47 · Python · PyTorch',
                    Icons.code_rounded, AppColors.textSecondary),
              ]),
              const SizedBox(height: 24),
              _buildSaveButton(),
              const SizedBox(height: 80),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return const Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Paramètres',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          )),
        SizedBox(height: 4),
        Text('Configuration de l\'application HiveSense',
          style: TextStyle(
            color: AppColors.textSecondary,
            fontSize: 12,
          )),
      ],
    );
  }

  // ── Section container ─────────────────────────────────────────────────────
  Widget _buildSection(String title, IconData icon,
      List<Widget> children) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                Icon(icon, color: AppColors.primary, size: 18),
                const SizedBox(width: 8),
                Text(title,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  )),
              ],
            ),
          ),
          const Divider(height: 1, color: AppColors.surfaceLight),
          ...children,
        ],
      ),
    );
  }

  // ── Switch tile ───────────────────────────────────────────────────────────
  Widget _switchTile(String title, String subtitle,
      bool value, ValueChanged<bool> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(
          horizontal: 14, vertical: 8),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 13,
                  )),
                Text(subtitle,
                  style: const TextStyle(
                    color: AppColors.textHint,
                    fontSize: 11,
                  )),
              ],
            ),
          ),
          Switch(
            value: value,
            onChanged: onChanged,
            activeColor: AppColors.primary,
          ),
        ],
      ),
    );
  }

  // ── Slider tile ───────────────────────────────────────────────────────────
  Widget _sliderTile(String title, String valueLabel,
      double value, double min, double max,
      ValueChanged<double> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(
          horizontal: 14, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(title,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 13,
                )),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 3),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(valueLabel,
                  style: const TextStyle(
                    color: AppColors.primary,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  )),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: AppColors.primary,
              inactiveTrackColor: AppColors.surfaceLight,
              thumbColor: AppColors.primary,
              overlayColor: AppColors.primary.withOpacity(0.1),
              trackHeight: 3,
            ),
            child: Slider(
              value: value,
              min: min,
              max: max,
              onChanged: onChanged,
            ),
          ),
        ],
      ),
    );
  }

  // ── Info tile ─────────────────────────────────────────────────────────────
  Widget _infoTile(String title, String subtitle,
      IconData icon, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(
          horizontal: 14, vertical: 10),
      child: Row(
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 13,
                  )),
                Text(subtitle,
                  style: const TextStyle(
                    color: AppColors.textHint,
                    fontSize: 11,
                  )),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── URL tile ──────────────────────────────────────────────────────────────
  Widget _urlTile(String label, String value,
      ValueChanged<String> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(
          horizontal: 14, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
            style: const TextStyle(
              color: AppColors.textSecondary,
              fontSize: 12,
            )),
          const SizedBox(height: 6),
          TextFormField(
            initialValue: value,
            onChanged: onChanged,
            style: const TextStyle(
              color: AppColors.textPrimary,
              fontSize: 13,
            ),
            decoration: InputDecoration(
              filled: true,
              fillColor: AppColors.surfaceLight,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide.none,
              ),
              contentPadding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 10),
              prefixIcon: const Icon(Icons.link_rounded,
                  color: AppColors.textHint, size: 18),
            ),
          ),
        ],
      ),
    );
  }

  // ── Action tile ───────────────────────────────────────────────────────────
  Widget _actionTile(String title, String subtitle,
      IconData icon, Color color, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(
            horizontal: 14, vertical: 12),
        child: Row(
          children: [
            Icon(icon, color: color, size: 18),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                    style: TextStyle(
                      color: color,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    )),
                  Text(subtitle,
                    style: const TextStyle(
                      color: AppColors.textHint,
                      fontSize: 11,
                    )),
                ],
              ),
            ),
            Icon(Icons.chevron_right_rounded,
                color: color, size: 18),
          ],
        ),
      ),
    );
  }

  // ── Bouton sauvegarder ────────────────────────────────────────────────────
  Widget _buildSaveButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: _saveSettings,
        icon: const Icon(Icons.save_rounded, size: 18),
        label: const Text('Sauvegarder les paramètres'),
      ),
    );
  }

  // ── Actions ───────────────────────────────────────────────────────────────
 Future<void> _testConnection() async {
  showDialog(
    context: context,
    barrierDismissible: false,
    builder: (_) => const AlertDialog(
      backgroundColor: AppColors.surface,
      content: Row(
        children: [
          CircularProgressIndicator(color: AppColors.primary),
          SizedBox(width: 16),
          Text('Test en cours...',
            style: TextStyle(color: AppColors.textPrimary)),
        ],
      ),
    ),
  );

  bool model1Ok = false;
  bool model2Ok = false;
  String model1Msg = '';
  String model2Msg = '';

  try {
    final resp = await http.get(
      Uri.parse('$_model1Url/health'),
    ).timeout(const Duration(seconds: 5));
    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body);
      model1Ok  = true;
      model1Msg = 'Val Acc : ${data['val_acc']}';
    }
  } catch (_) {
    model1Msg = 'Non joignable';
  }

  try {
    final resp = await http.get(
      Uri.parse('$_model2Url/health'),
    ).timeout(const Duration(seconds: 5));
    model2Ok  = resp.statusCode == 200;
    model2Msg = model2Ok ? 'Connecté' : 'Erreur';
  } catch (_) {
    model2Msg = 'Non joignable';
  }

  if (!mounted) return;

  // ⚠️ Fermer le dialog de chargement correctement
  Navigator.of(context, rootNavigator: true).pop();

  showDialog(
    context: context,
    builder: (_) => AlertDialog(
      backgroundColor: AppColors.surface,
      title: const Row(
        children: [
          Icon(Icons.wifi_tethering_rounded,
              color: AppColors.primary, size: 20),
          SizedBox(width: 8),
          Text('Résultat du test',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            )),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _connectionResult(
            'Modèle 1 — État de la reine',
            model1Ok, model1Msg),
          const SizedBox(height: 12),
          _connectionResult(
            'Modèle 2 — Anomalie acoustique',
            model2Ok, model2Msg),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () =>
            // ⚠️ Fermer le dialog de résultat correctement
            Navigator.of(context, rootNavigator: true).pop(),
          child: const Text('Fermer',
            style: TextStyle(color: AppColors.primary)),
        ),
      ],
    ),
  );
}

Widget _connectionResult(String label, bool ok, String msg) {
  return Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: ok
          ? AppColors.success.withOpacity(0.1)
          : AppColors.error.withOpacity(0.1),
      borderRadius: BorderRadius.circular(10),
      border: Border.all(
        color: ok
            ? AppColors.success.withOpacity(0.3)
            : AppColors.error.withOpacity(0.3),
      ),
    ),
    child: Row(
      children: [
        Icon(
          ok ? Icons.check_circle_rounded : Icons.error_rounded,
          color: ok ? AppColors.success : AppColors.error,
          size: 20,
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                )),
              Text(msg,
                style: TextStyle(
                  color: ok
                      ? AppColors.success
                      : AppColors.error,
                  fontSize: 11,
                )),
            ],
          ),
        ),
      ],
    ),
  );
}

  void _saveSettings() {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: AppColors.primary,
        content: const Row(
          children: [
            Icon(Icons.check_rounded,
                color: Colors.black, size: 18),
            SizedBox(width: 8),
            Text('Paramètres sauvegardés',
              style: TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.bold,
              )),
          ],
        ),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}