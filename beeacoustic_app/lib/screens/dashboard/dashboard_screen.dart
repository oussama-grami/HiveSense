import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import '../../core/theme/app_theme.dart';
import '../../models/hive_model.dart';
import '../../providers/hive_provider.dart';
import '../../widgets/common/status_badge.dart';
import '../../widgets/common/empty_state.dart';
import 'package:intl/intl.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<HiveProvider>(
      builder: (context, provider, _) {
        if (provider.isLoading) {
          return const Scaffold(
            backgroundColor: AppColors.background,
            body: Center(
              child: CircularProgressIndicator(
                  color: AppColors.primary),
            ),
          );
        }
        return _DashboardView(provider: provider);
      },
    );
  }
}

class _DashboardView extends StatelessWidget {
  final HiveProvider provider;
  const _DashboardView({required this.provider});

  String _timeAgo(DateTime time) {
    final diff = DateTime.now().difference(time);
    if (diff.inMinutes < 60) return '${diff.inMinutes}min';
    if (diff.inHours < 24)   return '${diff.inHours}h';
    return '${diff.inDays}j';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: provider.refresh,
          color: AppColors.primary,
          child: CustomScrollView(
            slivers: [
              _buildAppBar(context),
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildSummaryCards(),
                      const SizedBox(height: 24),
                      _buildAlertsSection(context),
                      const SizedBox(height: 24),
                      _buildHivesSection(context),
                      const SizedBox(height: 80),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar(BuildContext context) {
    return SliverAppBar(
      floating: true,
      backgroundColor: AppColors.background,
      title: Row(
        children: [
          Container(
            width: 36, height: 36,
            decoration: BoxDecoration(
              color: AppColors.primary,
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.hexagon_rounded,
                color: Colors.black, size: 22),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('HiveSense',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                )),
              Text(
                DateFormat('EEEE d MMMM', 'fr_FR')
                    .format(DateTime.now()),
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 11,
                )),
            ],
          ),
          const Spacer(),
          // Indicateurs statut API
          Row(
            children: [
              _apiDot(provider.model1Online),
              const SizedBox(width: 4),
              _apiDot(provider.model2Online),
            ],
          ),
        ],
      ),
      actions: [
        Stack(
          children: [
            IconButton(
              icon: const Icon(Icons.notifications_rounded,
                  color: AppColors.textPrimary),
              onPressed: () => context.go('/alerts'),
            ),
            if (provider.unreadCount > 0)
              Positioned(
                right: 8, top: 8,
                child: Container(
                  width: 16, height: 16,
                  decoration: const BoxDecoration(
                    color: AppColors.error,
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      '${provider.unreadCount}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 9,
                        fontWeight: FontWeight.bold,
                      )),
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(width: 8),
        IconButton(
  icon: const Icon(Icons.account_circle_rounded,
      color: AppColors.primary),
  onPressed: () => context.go('/profile'),
),
      ],
    );
  }

  Widget _apiDot(bool online) {
    return Tooltip(
      message: online ? 'API connectée' : 'API hors ligne',
      child: Container(
        width: 8, height: 8,
        decoration: BoxDecoration(
          color: online ? AppColors.success : AppColors.hiveOffline,
          shape: BoxShape.circle,
        ),
      ),
    );
  }

  Widget _buildSummaryCards() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Vue d\'ensemble',
          style: TextStyle(
            color: AppColors.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          )),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(child: _summaryCard('Total',
                '${provider.totalHives}',
                Icons.hive_rounded, AppColors.info)),
            const SizedBox(width: 10),
            Expanded(child: _summaryCard('Normales',
                '${provider.normalHives}',
                Icons.check_circle_rounded, AppColors.success)),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(child: _summaryCard('Attention',
                '${provider.warningHives}',
                Icons.warning_rounded, AppColors.warning)),
            const SizedBox(width: 10),
            Expanded(child: _summaryCard('Critiques',
                '${provider.criticalHives}',
                Icons.error_rounded, AppColors.error)),
          ],
        ),
      ],
    );
  }

  Widget _summaryCard(String label, String value,
      IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            width: 42, height: 42,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(value,
                style: TextStyle(
                  color: color,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                )),
              Text(label,
                style: const TextStyle(
                  color: AppColors.textSecondary,
                  fontSize: 12,
                )),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAlertsSection(BuildContext context) {
    final alerts = provider.unreadAlerts.take(3).toList();
    if (alerts.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text('Alertes récentes',
              style: TextStyle(
                color: AppColors.textPrimary,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              )),
            TextButton(
              onPressed: () => context.go('/alerts'),
              child: const Text('Tout voir',
                style: TextStyle(color: AppColors.primary)),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ...alerts.map((alert) => _alertCard(alert)),
      ],
    );
  }

  Widget _alertCard(AlertModel alert) {
    final color = alert.type == AlertType.queenMissing
        ? AppColors.error
        : alert.type == AlertType.offline
            ? AppColors.hiveOffline
            : AppColors.warning;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(_alertIcon(alert.type), color: color, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(alert.hiveName,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  )),
                Text(alert.message,
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 12,
                  )),
              ],
            ),
          ),
          Text(_timeAgo(alert.timestamp),
            style: const TextStyle(
              color: AppColors.textHint,
              fontSize: 11,
            )),
        ],
      ),
    );
  }

  IconData _alertIcon(AlertType type) {
    switch (type) {
      case AlertType.queenMissing:    return Icons.error_rounded;
      case AlertType.offline:         return Icons.wifi_off_rounded;
      case AlertType.batteryLow:      return Icons.battery_alert_rounded;
      case AlertType.temperatureHigh: return Icons.thermostat_rounded;
      default:                        return Icons.warning_rounded;
    }
  }

  Widget _buildHivesSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text('Mes ruches',
              style: TextStyle(
                color: AppColors.textPrimary,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              )),
            TextButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.add_rounded,
                  color: AppColors.primary, size: 18),
              label: const Text('Ajouter',
                style: TextStyle(color: AppColors.primary)),
            ),
          ],
        ),
        const SizedBox(height: 8),
        if (provider.hives.isEmpty)
          EmptyState(
            icon:        Icons.hive_rounded,
            title:       'Aucune ruche',
            subtitle:    'Ajoutez votre première ruche pour commencer la surveillance.',
            buttonLabel: 'Ajouter une ruche',
            onButtonTap: () {},
          )
        else
          ...provider.hives.map((hive) =>
              _HoverCard(
                onTap: () => context.go('/hive/${hive.id}'),
                borderColor: hive.statusColor,
                child: _hiveCardContent(hive),
              )),
      ],
    );
  }

  Widget _hiveCardContent(HiveModel hive) {
    return Column(
      children: [
        Row(
          children: [
            Container(
              width: 46, height: 46,
              decoration: BoxDecoration(
                color: hive.statusColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(Icons.hive_rounded,
                  color: hive.statusColor, size: 26),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(hive.name,
                    style: const TextStyle(
                      color: AppColors.textPrimary,
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                    )),
                  Text(hive.location,
                    style: const TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 12,
                    )),
                ],
              ),
            ),
            StatusBadge(
              label:   hive.status,
              color:   hive.statusColor,
              icon:    hive.statusIcon,
              pulsing: hive.status == 'Critique',
            ),
          ],
        ),
        const SizedBox(height: 14),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _sensorChip(Icons.thermostat_rounded,
                '${hive.sensors.temperature}°C',
                _tempColor(hive.sensors.temperature)),
            _sensorChip(Icons.water_drop_rounded,
                '${hive.sensors.humidity}%', AppColors.info),
            _sensorChip(Icons.battery_charging_full_rounded,
                '${hive.sensors.batteryLevel}%',
                _batteryColor(hive.sensors.batteryLevel)),
            _sensorChip(Icons.network_cell_rounded,
                hive.isOnline ? 'En ligne' : 'Hors ligne',
                hive.isOnline
                    ? AppColors.success
                    : AppColors.hiveOffline),
          ],
        ),
        if (hive.queenPrediction != null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: AppColors.surfaceLight,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                const Icon(Icons.psychology_rounded,
                    color: AppColors.primary, size: 14),
                const SizedBox(width: 6),
                Text('IA : ${hive.queenPrediction!.className}',
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 12,
                  )),
                const Spacer(),
                Text(
                  '${(hive.queenPrediction!.confidence * 100)
                      .toStringAsFixed(0)}%',
                  style: const TextStyle(
                    color: AppColors.primary,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  )),
              ],
            ),
          ),
        ],
        const SizedBox(height: 8),
        Row(
          children: [
            const Icon(Icons.access_time_rounded,
                color: AppColors.textHint, size: 12),
            const SizedBox(width: 4),
            Text('Mis à jour ${_timeAgo(hive.lastUpdate)}',
              style: const TextStyle(
                color: AppColors.textHint, fontSize: 11)),
            const Spacer(),
            const Icon(Icons.chevron_right_rounded,
                color: AppColors.primary, size: 16),
          ],
        ),
      ],
    );
  }

  Widget _sensorChip(IconData icon, String value, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 16),
        const SizedBox(height: 2),
        Text(value,
          style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          )),
      ],
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
}

// ── HoverCard ─────────────────────────────────────────────────────────────────
class _HoverCard extends StatefulWidget {
  final Widget       child;
  final VoidCallback onTap;
  final Color        borderColor;

  const _HoverCard({
    required this.child,
    required this.onTap,
    required this.borderColor,
  });

  @override
  State<_HoverCard> createState() => _HoverCardState();
}

class _HoverCardState extends State<_HoverCard> {
  bool _isHovered = false;
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter:  (_) => setState(() => _isHovered = true),
      onExit:   (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTap:       widget.onTap,
        onTapDown:   (_) => setState(() => _isPressed = true),
        onTapUp:     (_) => setState(() => _isPressed = false),
        onTapCancel: () => setState(() => _isPressed = false),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          transform: Matrix4.identity()
            ..scale(_isPressed ? 0.97 : 1.0),
          decoration: BoxDecoration(
            color: _isHovered
                ? AppColors.surfaceLight
                : AppColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: _isHovered
                  ? widget.borderColor.withOpacity(0.7)
                  : widget.borderColor.withOpacity(0.3),
              width: _isHovered ? 1.5 : 1,
            ),
            boxShadow: _isHovered
                ? [BoxShadow(
                    color: widget.borderColor.withOpacity(0.15),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  )]
                : [],
          ),
          child: widget.child,
        ),
      ),
    );
  }
}