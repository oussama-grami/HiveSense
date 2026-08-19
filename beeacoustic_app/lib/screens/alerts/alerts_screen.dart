import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../../models/hive_model.dart';
import '../../providers/hive_provider.dart';

class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});
  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen> {
  AlertType? _selectedFilter;

  List<AlertModel> _filteredAlerts(HiveProvider provider) {
    final alerts = provider.allAlerts;
    if (_selectedFilter != null) {
      return alerts.where((a) => a.type == _selectedFilter).toList();
    }
    return alerts;
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<HiveProvider>(
      builder: (context, provider, _) {
        final alerts = _filteredAlerts(provider);
        return Scaffold(
          backgroundColor: AppColors.background,
          body: SafeArea(
            child: Column(
              children: [
                _buildHeader(context, provider),
                _buildFilters(),
                Expanded(
                  child: alerts.isEmpty
                      ? _buildEmpty()
                      : _buildAlertsList(alerts, provider),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildHeader(BuildContext context, HiveProvider provider) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Alertes',
                  style: TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  )),
                Text('${provider.unreadCount} non lue(s)',
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 12,
                  )),
              ],
            ),
          ),
          if (provider.unreadCount > 0)
            TextButton.icon(
              onPressed: provider.markAllAlertsRead,
              icon: const Icon(Icons.done_all_rounded,
                  color: AppColors.primary, size: 16),
              label: const Text('Tout lire',
                style: TextStyle(
                  color: AppColors.primary, fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Widget _buildFilters() {
    final filters = [
      (null,                     'Tout',          Icons.all_inbox_rounded),
      (AlertType.queenMissing,   'Reine absente', Icons.error_rounded),
      (AlertType.offline,        'Hors ligne',    Icons.wifi_off_rounded),
      (AlertType.batteryLow,     'Batterie',      Icons.battery_alert_rounded),
      (AlertType.temperatureHigh,'Température',   Icons.thermostat_rounded),
    ];
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: filters.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final (type, label, icon) = filters[i];
          final isSelected = _selectedFilter == type;
          return GestureDetector(
            onTap: () => setState(() => _selectedFilter = type),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 150),
              padding: const EdgeInsets.symmetric(
                  horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: isSelected
                    ? AppColors.primary : AppColors.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: isSelected
                      ? AppColors.primary : AppColors.surfaceLight),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon,
                    color: isSelected
                        ? Colors.black : AppColors.textSecondary,
                    size: 14),
                  const SizedBox(width: 6),
                  Text(label,
                    style: TextStyle(
                      color: isSelected
                          ? Colors.black : AppColors.textSecondary,
                      fontSize: 12,
                      fontWeight: isSelected
                          ? FontWeight.bold : FontWeight.normal,
                    )),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildAlertsList(
      List<AlertModel> alerts, HiveProvider provider) {
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: alerts.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (_, i) =>
          _alertCard(alerts[i], provider),
    );
  }

  Widget _alertCard(AlertModel alert, HiveProvider provider) {
    final color = _alertColor(alert.type);
    return GestureDetector(
      onTap: () => provider.markAlertRead(alert.id),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: alert.isRead
              ? AppColors.surface : color.withOpacity(0.08),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: alert.isRead
                ? AppColors.surfaceLight : color.withOpacity(0.4)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 40, height: 40,
              decoration: BoxDecoration(
                color: color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(_alertIcon(alert.type),
                  color: color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(alert.hiveName,
                          style: TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 14,
                            fontWeight: alert.isRead
                                ? FontWeight.normal
                                : FontWeight.bold,
                          ))),
                      Text(_timeAgo(alert.timestamp),
                        style: const TextStyle(
                          color: AppColors.textHint,
                          fontSize: 11,
                        )),
                    ],
                  ),
                  const SizedBox(height: 3),
                  Text(alert.message,
                    style: const TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 12,
                    )),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(_alertLabel(alert.type),
                      style: TextStyle(
                        color: color,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      )),
                  ),
                ],
              ),
            ),
            if (!alert.isRead)
              Container(
                width: 8, height: 8,
                margin: const EdgeInsets.only(top: 4),
                decoration: BoxDecoration(
                  color: color, shape: BoxShape.circle),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 80, height: 80,
            decoration: const BoxDecoration(
              color: AppColors.surface, shape: BoxShape.circle),
            child: const Icon(Icons.notifications_off_rounded,
                color: AppColors.textHint, size: 40),
          ),
          const SizedBox(height: 16),
          const Text('Aucune alerte',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            )),
          const SizedBox(height: 4),
          const Text('Toutes vos ruches sont en bon état',
            style: TextStyle(
              color: AppColors.textSecondary, fontSize: 13)),
        ],
      ),
    );
  }

  Color _alertColor(AlertType type) {
    switch (type) {
      case AlertType.queenMissing:    return AppColors.error;
      case AlertType.queenRejected:   return AppColors.warning;
      case AlertType.offline:         return AppColors.hiveOffline;
      case AlertType.batteryLow:      return AppColors.warning;
      case AlertType.temperatureHigh: return AppColors.error;
      case AlertType.temperatureLow:  return AppColors.info;
      default:                        return AppColors.info;
    }
  }

  IconData _alertIcon(AlertType type) {
    switch (type) {
      case AlertType.queenMissing:    return Icons.error_rounded;
      case AlertType.queenRejected:   return Icons.warning_rounded;
      case AlertType.offline:         return Icons.wifi_off_rounded;
      case AlertType.batteryLow:      return Icons.battery_alert_rounded;
      case AlertType.temperatureHigh: return Icons.thermostat_rounded;
      default:                        return Icons.notifications_rounded;
    }
  }

  String _alertLabel(AlertType type) {
    switch (type) {
      case AlertType.queenMissing:    return 'REINE ABSENTE';
      case AlertType.queenRejected:   return 'REINE REJETÉE';
      case AlertType.offline:         return 'HORS LIGNE';
      case AlertType.batteryLow:      return 'BATTERIE FAIBLE';
      case AlertType.temperatureHigh: return 'TEMPÉRATURE ÉLEVÉE';
      default:                        return 'ALERTE';
    }
  }

  String _timeAgo(DateTime time) {
    final diff = DateTime.now().difference(time);
    if (diff.inMinutes < 60) return '${diff.inMinutes}min';
    if (diff.inHours < 24)   return '${diff.inHours}h';
    return '${diff.inDays}j';
  }
}