import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../core/theme/app_theme.dart';
import '../../models/hive_model.dart';

class StatisticsScreen extends StatefulWidget {
  const StatisticsScreen({super.key});
  @override
  State<StatisticsScreen> createState() => _StatisticsScreenState();
}

class _StatisticsScreenState extends State<StatisticsScreen> {
  final List<HiveModel> _hives = HiveModel.getMockHives();
  int _selectedPeriod = 7;
  String _selectedHive = 'Toutes';

  // Données simulées
  final Map<String, List<double>> _tempData = {
    'Ruche Alpha': [33.5, 34.1, 34.8, 33.9, 34.2, 35.1, 34.2],
    'Ruche Beta':  [35.1, 35.8, 36.2, 35.5, 35.8, 36.5, 35.8],
    'Ruche Gamma': [33.0, 33.2, 33.5, 33.1, 33.0, 33.4, 33.0],
  };

  final Map<String, List<double>> _humidityData = {
    'Ruche Alpha': [60.0, 63.2, 61.5, 64.0, 62.5, 65.0, 62.5],
    'Ruche Beta':  [68.0, 71.0, 70.5, 72.0, 71.0, 73.0, 71.0],
    'Ruche Gamma': [63.0, 65.0, 64.0, 66.0, 65.0, 67.0, 65.0],
  };

  final List<String> _days = ['L', 'M', 'M', 'J', 'V', 'S', 'D'];

  final List<Color> _hiveColors = [
    AppColors.primary,
    AppColors.error,
    AppColors.info,
  ];

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
              const SizedBox(height: 16),
              _buildFilters(),
              const SizedBox(height: 24),
              _buildKPICards(),
              const SizedBox(height: 24),
              _buildTemperatureChart(),
              const SizedBox(height: 20),
              _buildHumidityChart(),
              const SizedBox(height: 20),
              _buildStatusDistribution(),
              const SizedBox(height: 20),
              _buildActivityHeatmap(),
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
        Text('Statistiques',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 24,
            fontWeight: FontWeight.bold,
          )),
        SizedBox(height: 4),
        Text('Analyse des données de vos ruches',
          style: TextStyle(
            color: AppColors.textSecondary,
            fontSize: 12,
          )),
      ],
    );
  }

  // ── Filtres ───────────────────────────────────────────────────────────────
  Widget _buildFilters() {
    return Row(
      children: [
        // Filtre période
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 12, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(12),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<int>(
                value: _selectedPeriod,
                dropdownColor: AppColors.surface,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 13,
                ),
                items: const [
                  DropdownMenuItem(value: 7,  child: Text('7 derniers jours')),
                  DropdownMenuItem(value: 14, child: Text('14 derniers jours')),
                  DropdownMenuItem(value: 30, child: Text('30 derniers jours')),
                ],
                onChanged: (v) =>
                    setState(() => _selectedPeriod = v ?? 7),
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        // Filtre ruche
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 12, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(12),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: _selectedHive,
                dropdownColor: AppColors.surface,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 13,
                ),
                items: [
                  const DropdownMenuItem(
                      value: 'Toutes', child: Text('Toutes les ruches')),
                  ..._hives.map((h) => DropdownMenuItem(
                      value: h.name, child: Text(h.name))),
                ],
                onChanged: (v) =>
                    setState(() => _selectedHive = v ?? 'Toutes'),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ── KPI Cards ─────────────────────────────────────────────────────────────
  Widget _buildKPICards() {
    final avgTemp = _hives
        .map((h) => h.sensors.temperature)
        .reduce((a, b) => a + b) / _hives.length;
    final avgHumidity = _hives
        .map((h) => h.sensors.humidity)
        .reduce((a, b) => a + b) / _hives.length;
    final onlineCount = _hives.where((h) => h.isOnline).length;
    final totalAlerts = _hives
        .expand((h) => h.recentAlerts)
        .length;

    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.6,
      children: [
        _kpiCard('Temp. moyenne',
            '${avgTemp.toStringAsFixed(1)}°C',
            Icons.thermostat_rounded,
            AppColors.warning),
        _kpiCard('Humidité moy.',
            '${avgHumidity.toStringAsFixed(1)}%',
            Icons.water_drop_rounded,
            AppColors.info),
        _kpiCard('Ruches en ligne',
            '$onlineCount/${_hives.length}',
            Icons.wifi_rounded,
            AppColors.success),
        _kpiCard('Alertes totales',
            '$totalAlerts',
            Icons.notifications_rounded,
            AppColors.error),
      ],
    );
  }

  Widget _kpiCard(String label, String value,
      IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Container(
            width: 38, height: 38,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(value,
                  style: TextStyle(
                    color: color,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  )),
                Text(label,
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 10,
                  )),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Graphique température ─────────────────────────────────────────────────
  Widget _buildTemperatureChart() {
    return _chartContainer(
      title: 'Température (°C)',
      icon: Icons.thermostat_rounded,
      color: AppColors.warning,
      child: _multiLineChart(_tempData, 30, 38, '°C'),
    );
  }

  // ── Graphique humidité ────────────────────────────────────────────────────
  Widget _buildHumidityChart() {
    return _chartContainer(
      title: 'Humidité (%)',
      icon: Icons.water_drop_rounded,
      color: AppColors.info,
      child: _multiLineChart(_humidityData, 50, 80, '%'),
    );
  }

  Widget _chartContainer({
    required String title,
    required IconData icon,
    required Color color,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 8),
              Text(title,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                )),
            ],
          ),
          const SizedBox(height: 6),
          // Légende
          if (_selectedHive == 'Toutes')
            Wrap(
              spacing: 12,
              children: _hives.asMap().entries.map((e) =>
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 10, height: 10,
                      decoration: BoxDecoration(
                        color: _hiveColors[e.key],
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(e.value.name,
                      style: const TextStyle(
                        color: AppColors.textHint,
                        fontSize: 10,
                      )),
                  ],
                ),
              ).toList(),
            ),
          const SizedBox(height: 12),
          SizedBox(height: 160, child: child),
        ],
      ),
    );
  }

  Widget _multiLineChart(Map<String, List<double>> data,
      double minY, double maxY, String unit) {
    final hivesToShow = _selectedHive == 'Toutes'
        ? data.entries.toList()
        : data.entries
            .where((e) => e.key == _selectedHive)
            .toList();

    return LineChart(
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
              reservedSize: 38,
              getTitlesWidget: (val, _) => Text(
                '${val.toInt()}$unit',
                style: const TextStyle(
                  color: AppColors.textHint,
                  fontSize: 9,
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
        lineBarsData: hivesToShow.asMap().entries.map((entry) {
          final color = _hiveColors[
              data.keys.toList().indexOf(entry.value.key) %
                  _hiveColors.length];
          return LineChartBarData(
            spots: entry.value.value.asMap().entries.map((e) =>
              FlSpot(e.key.toDouble(), e.value)).toList(),
            isCurved: true,
            color: color,
            barWidth: 2,
            dotData: FlDotData(
              getDotPainter: (_, __, ___, ____) =>
                  FlDotCirclePainter(
                radius: 3,
                color: color,
                strokeWidth: 1.5,
                strokeColor: AppColors.surface,
              ),
            ),
            belowBarData: BarAreaData(
              show: hivesToShow.length == 1,
              color: color.withOpacity(0.1),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Distribution des statuts ──────────────────────────────────────────────
  Widget _buildStatusDistribution() {
    final normal   = _hives.where((h) => h.status == 'Normal').length;
    final warning  = _hives.where((h) => h.status == 'Attention').length;
    final critical = _hives.where((h) => h.status == 'Critique').length;
    final offline  = _hives.where((h) => h.status == 'Hors ligne').length;
    final total    = _hives.length;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.pie_chart_rounded,
                  color: AppColors.primary, size: 18),
              SizedBox(width: 8),
              Text('Distribution des statuts',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                )),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              SizedBox(
                height: 120, width: 120,
                child: PieChart(
                  PieChartData(
                    sectionsSpace: 2,
                    centerSpaceRadius: 30,
                    sections: [
                      if (normal > 0) PieChartSectionData(
                        value: normal.toDouble(),
                        color: AppColors.success,
                        radius: 30,
                        showTitle: false,
                      ),
                      if (warning > 0) PieChartSectionData(
                        value: warning.toDouble(),
                        color: AppColors.warning,
                        radius: 30,
                        showTitle: false,
                      ),
                      if (critical > 0) PieChartSectionData(
                        value: critical.toDouble(),
                        color: AppColors.error,
                        radius: 30,
                        showTitle: false,
                      ),
                      if (offline > 0) PieChartSectionData(
                        value: offline.toDouble(),
                        color: AppColors.hiveOffline,
                        radius: 30,
                        showTitle: false,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  children: [
                    _legendItem('Normales', normal,
                        total, AppColors.success),
                    _legendItem('Attention', warning,
                        total, AppColors.warning),
                    _legendItem('Critiques', critical,
                        total, AppColors.error),
                    _legendItem('Hors ligne', offline,
                        total, AppColors.hiveOffline),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _legendItem(String label, int count,
      int total, Color color) {
    final pct = total > 0
        ? (count / total * 100).toStringAsFixed(0)
        : '0';
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
            width: 10, height: 10,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(label,
              style: const TextStyle(
                color: AppColors.textSecondary,
                fontSize: 12,
              )),
          ),
          Text('$count ($pct%)',
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            )),
        ],
      ),
    );
  }

  // ── Heatmap activité ──────────────────────────────────────────────────────
  Widget _buildActivityHeatmap() {
    final hours = List.generate(24, (i) => i);
    final activityLevels = [
      0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
      0.4, 0.7, 0.9, 1.0, 0.9, 0.8,
      0.7, 0.8, 0.9, 0.8, 0.7, 0.6,
      0.5, 0.4, 0.3, 0.2, 0.1, 0.1,
    ];

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.access_time_rounded,
                  color: AppColors.primary, size: 18),
              SizedBox(width: 8),
              Text('Activité acoustique par heure',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                )),
            ],
          ),
          const SizedBox(height: 4),
          const Text('Intensité moyenne du bourdonnement (24h)',
            style: TextStyle(
              color: AppColors.textHint,
              fontSize: 11,
            )),
          const SizedBox(height: 16),
          SizedBox(
            height: 50,
            child: Row(
              children: hours.map((h) {
                final level = activityLevels[h];
                return Expanded(
                  child: Tooltip(
                    message: '${h}h : ${(level * 100).toInt()}%',
                    child: Container(
                      margin: const EdgeInsets.symmetric(horizontal: 1),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(
                            0.1 + level * 0.9),
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: ['0h', '6h', '12h', '18h', '23h'].map((t) =>
              Text(t,
                style: const TextStyle(
                  color: AppColors.textHint,
                  fontSize: 10,
                ))).toList(),
          ),
        ],
      ),
    );
  }
}