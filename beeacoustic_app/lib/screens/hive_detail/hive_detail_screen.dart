import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../core/theme/app_theme.dart';
import '../../models/hive_model.dart';

class HiveDetailScreen extends StatefulWidget {
  final String hiveId;
  const HiveDetailScreen({super.key, required this.hiveId});

  @override
  State<HiveDetailScreen> createState() => _HiveDetailScreenState();
}

class _HiveDetailScreenState extends State<HiveDetailScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  late HiveModel _hive;

  // Données simulées pour les graphiques (7 derniers jours)
  final List<double> _tempHistory = [
    33.5, 34.1, 34.8, 33.9, 34.2, 35.1, 34.2];
  final List<double> _humidityHistory = [
    60.0, 63.2, 61.5, 64.0, 62.5, 65.0, 62.5];
  final List<String> _days = ['L', 'M', 'M', 'J', 'V', 'S', 'D'];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _hive = HiveModel.getMockHives()
        .firstWhere((h) => h.id == widget.hiveId,
            orElse: () => HiveModel.getMockHives().first);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildTabBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildSensorsTab(),
                  _buildAITab(),
                  _buildHistoryTab(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          GestureDetector(
            onTap: () => context.go('/'),
            child: Container(
              width: 36, height: 36,
              decoration: BoxDecoration(
                color: AppColors.surfaceLight,
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.arrow_back_rounded,
                  color: AppColors.textPrimary, size: 20),
            ),
          ),
          const SizedBox(width: 12),
          Container(
            width: 44, height: 44,
            decoration: BoxDecoration(
              color: _hive.statusColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(Icons.hive_rounded,
                color: _hive.statusColor, size: 26),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(_hive.name,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  )),
                Text(_hive.location,
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 12,
                  )),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: _hive.statusColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(_hive.statusIcon,
                    color: _hive.statusColor, size: 12),
                const SizedBox(width: 4),
                Text(_hive.status,
                  style: TextStyle(
                    color: _hive.statusColor,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  )),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── TabBar ────────────────────────────────────────────────────────────────
  Widget _buildTabBar() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
      ),
      child: TabBar(
        controller: _tabController,
        indicator: BoxDecoration(
          color: AppColors.primary,
          borderRadius: BorderRadius.circular(10),
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        labelColor: Colors.black,
        unselectedLabelColor: AppColors.textSecondary,
        labelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontSize: 12),
        tabs: const [
          Tab(text: 'Capteurs'),
          Tab(text: 'Intelligence IA'),
          Tab(text: 'Historique'),
        ],
      ),
    );
  }

  // ── Onglet Capteurs ───────────────────────────────────────────────────────
  Widget _buildSensorsTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Température et humidité
          Row(
            children: [
              Expanded(child: _bigSensorCard(
                'Température\nRuche',
                '${_hive.sensors.temperature}°C',
                Icons.thermostat_rounded,
                _tempColor(_hive.sensors.temperature),
                'Optimal : 33-36°C',
              )),
              const SizedBox(width: 12),
              Expanded(child: _bigSensorCard(
                'Humidité\nRuche',
                '${_hive.sensors.humidity}%',
                Icons.water_drop_rounded,
                AppColors.info,
                'Optimal : 50-80%',
              )),
            ],
          ),
          const SizedBox(height: 12),
          // Météo externe
          _sectionTitle('Conditions extérieures'),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(child: _smallSensorCard(
                'Temp. extérieure',
                '${_hive.sensors.weatherTemp}°C',
                Icons.wb_sunny_rounded,
                AppColors.warning,
              )),
              const SizedBox(width: 10),
              Expanded(child: _smallSensorCard(
                'Humidité ext.',
                '${_hive.sensors.weatherHumidity}%',
                Icons.cloud_rounded,
                AppColors.info,
              )),
              const SizedBox(width: 10),
              Expanded(child: _smallSensorCard(
                'Vent',
                '${_hive.sensors.windSpeed} m/s',
                Icons.air_rounded,
                AppColors.textSecondary,
              )),
            ],
          ),
          const SizedBox(height: 12),
          // Device IoT
          _sectionTitle('Device IoT'),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(child: _smallSensorCard(
                'Batterie',
                '${_hive.sensors.batteryLevel}%',
                Icons.battery_charging_full_rounded,
                _batteryColor(_hive.sensors.batteryLevel),
              )),
              const SizedBox(width: 10),
              Expanded(child: _smallSensorCard(
                'Signal 4G',
                '${_hive.sensors.signalStrength} dBm',
                Icons.network_cell_rounded,
                _signalColor(_hive.sensors.signalStrength),
              )),
              const SizedBox(width: 10),
              Expanded(child: _smallSensorCard(
                'Statut',
                _hive.isOnline ? 'En ligne' : 'Hors ligne',
                Icons.wifi_rounded,
                _hive.isOnline
                    ? AppColors.success
                    : AppColors.hiveOffline,
              )),
            ],
          ),
          const SizedBox(height: 12),
          // Pression
          _sectionTitle('Pression atmosphérique'),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              children: [
                const Icon(Icons.compress_rounded,
                    color: AppColors.primary, size: 28),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('${_hive.sensors.pressure} hPa',
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                      )),
                    const Text('Pression normale',
                      style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 12,
                      )),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  // ── Onglet IA ─────────────────────────────────────────────────────────────
  Widget _buildAITab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Modèle 1
          _sectionTitle('Modèle 1 — État de la reine'),
          const SizedBox(height: 8),
          if (_hive.queenPrediction != null)
            _aiPredictionCard(_hive.queenPrediction!, AppColors.primary),
          const SizedBox(height: 16),
          // Modèle 2
          _sectionTitle('Modèle 2 — Anomalie acoustique'),
          const SizedBox(height: 8),
          if (_hive.anomalyPrediction != null)
            _aiPredictionCard(
                _hive.anomalyPrediction!, AppColors.secondaryLight)
          else
            _aiNotAvailable(),
          const SizedBox(height: 16),
          // Bouton analyser
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () => context.go('/audio'),
              icon: const Icon(Icons.mic_rounded),
              label: const Text('Lancer une nouvelle analyse'),
            ),
          ),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _aiPredictionCard(AIPrediction pred, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Résultat principal
          Row(
            children: [
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(Icons.psychology_rounded,
                    color: color, size: 28),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(pred.className,
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      )),
                    Text('Confiance : ${(pred.confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(color: color, fontSize: 13)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Barre de confiance
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: pred.confidence,
              backgroundColor: AppColors.surfaceLight,
              valueColor: AlwaysStoppedAnimation<Color>(color),
              minHeight: 6,
            ),
          ),
          const SizedBox(height: 12),
          // Recommandation
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppColors.surfaceLight,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              children: [
                const Icon(Icons.info_rounded,
                    color: AppColors.primary, size: 16),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(pred.recommendation,
                    style: const TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 12,
                    )),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          // Probabilités
          const Text('Distribution des probabilités',
            style: TextStyle(
              color: AppColors.textSecondary,
              fontSize: 12,
            )),
          const SizedBox(height: 8),
          ...pred.probabilities.entries.map((e) => Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(e.key,
                      style: const TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 11,
                      )),
                    Text('${(e.value * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        color: color,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      )),
                  ],
                ),
                const SizedBox(height: 2),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: e.value,
                    backgroundColor: AppColors.surfaceLight,
                    valueColor:
                        AlwaysStoppedAnimation<Color>(color),
                    minHeight: 4,
                  ),
                ),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _aiNotAvailable() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.surfaceLight),
      ),
      child: const Column(
        children: [
          Icon(Icons.hourglass_empty_rounded,
              color: AppColors.textHint, size: 36),
          SizedBox(height: 8),
          Text('Modèle 2 en cours de développement',
            style: TextStyle(
              color: AppColors.textSecondary,
              fontSize: 13,
            )),
          Text('Disponible prochainement',
            style: TextStyle(
              color: AppColors.textHint,
              fontSize: 11,
            )),
        ],
      ),
    );
  }

  // ── Onglet Historique ─────────────────────────────────────────────────────
  Widget _buildHistoryTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _sectionTitle('Température — 7 derniers jours'),
          const SizedBox(height: 8),
          _lineChart(_tempHistory, AppColors.error, '°C', 30, 38),
          const SizedBox(height: 20),
          _sectionTitle('Humidité — 7 derniers jours'),
          const SizedBox(height: 8),
          _lineChart(_humidityHistory, AppColors.info, '%', 50, 80),
          const SizedBox(height: 80),
        ],
      ),
    );
  }

  Widget _lineChart(List<double> data, Color color,
      String unit, double minY, double maxY) {
    return Container(
      height: 180,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: LineChart(
        LineChartData(
          minY: minY,
          maxY: maxY,
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            getDrawingHorizontalLine: (_) => FlLine(
              color: AppColors.surfaceLight,
              strokeWidth: 1,
            ),
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 36,
                getTitlesWidget: (val, _) => Text(
                  '${val.toInt()}$unit',
                  style: const TextStyle(
                    color: AppColors.textHint,
                    fontSize: 10,
                  )),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (val, _) => Text(
                  _days[val.toInt() % _days.length],
                  style: const TextStyle(
                    color: AppColors.textHint,
                    fontSize: 10,
                  )),
              ),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false)),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: data.asMap().entries.map((e) =>
                FlSpot(e.key.toDouble(), e.value)).toList(),
              isCurved: true,
              color: color,
              barWidth: 2.5,
              dotData: FlDotData(
                getDotPainter: (_, __, ___, ____) =>
                    FlDotCirclePainter(
                  radius: 4,
                  color: color,
                  strokeWidth: 2,
                  strokeColor: AppColors.surface,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                color: color.withOpacity(0.1),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _sectionTitle(String title) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Text(title,
        style: const TextStyle(
          color: AppColors.textPrimary,
          fontSize: 15,
          fontWeight: FontWeight.bold,
        )),
    );
  }

  Widget _bigSensorCard(String label, String value,
      IconData icon, Color color, String subtitle) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(height: 8),
          Text(value,
            style: TextStyle(
              color: color,
              fontSize: 26,
              fontWeight: FontWeight.bold,
            )),
          Text(label,
            style: const TextStyle(
              color: AppColors.textSecondary,
              fontSize: 12,
            )),
          const SizedBox(height: 4),
          Text(subtitle,
            style: const TextStyle(
              color: AppColors.textHint,
              fontSize: 10,
            )),
        ],
      ),
    );
  }

  Widget _smallSensorCard(String label, String value,
      IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(height: 6),
          Text(value,
            style: TextStyle(
              color: color,
              fontSize: 13,
              fontWeight: FontWeight.bold,
            )),
          Text(label,
            style: const TextStyle(
              color: AppColors.textHint,
              fontSize: 10,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Color _tempColor(double temp) {
    if (temp > 36) return AppColors.error;
    if (temp < 32) return AppColors.info;
    return AppColors.success;
  }

  Color _batteryColor(int level) {
    if (level < 20) return AppColors.error;
    if (level < 40) return AppColors.warning;
    return AppColors.success;
  }

  Color _signalColor(int signal) {
    if (signal > -70) return AppColors.success;
    if (signal > -85) return AppColors.warning;
    return AppColors.error;
  }
}