#!/bin/bash
set -e

BROKER="kafka:9092"
PARTITIONS=3
REPLICATION=1
RETENTION_MS=604800000

wait_kafka() {
    echo "Ожидание готовности Kafka..."
    until kafka-topics --bootstrap-server "$BROKER" --list > /dev/null 2>&1; do
        sleep 2
    done
    echo "Kafka готова."
}

create_topic() {
    local name=$1
    local partitions=${2:-$PARTITIONS}
    echo "Создание топика: $name"
    kafka-topics --bootstrap-server "$BROKER" \
        --create \
        --if-not-exists \
        --topic "$name" \
        --partitions "$partitions" \
        --replication-factor "$REPLICATION" \
        --config retention.ms="$RETENTION_MS" \
        --config min.insync.replicas=1
}

wait_kafka

create_topic parse_jobs       3
create_topic cluster_jobs     3
create_topic analysis_done    3
create_topic auto_reply_jobs  3
create_topic parse_jobs.DLT   1
create_topic cluster_jobs.DLT 1
create_topic auto_reply_jobs.DLT 1

echo "Все топики успешно созданы:"
kafka-topics --bootstrap-server "$BROKER" --list
