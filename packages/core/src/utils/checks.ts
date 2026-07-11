/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/* Fail to compile on unexpected values. */
export function assumeExhaustive(_value: never): void {}

/**
 * Throws an exception on unexpected values.
 *
 * A common use case is switch statements:
 * switch(enumValue) {
 *   case Enum.A:
 *   case Enum.B:
 *     break;
 *   default:
 *     checkExhaustive(enumValue);
 * }
 */
export function checkExhaustive(
  value: never,
  msg = `unexpected value ${value}!`,
): never {
  assumeExhaustive(value);
  throw new Error(msg);
}

/**
 * Safely checks if an object has a property.
 */
export function hasProperty<T extends string>(
  obj: unknown,
  prop: T,
): obj is { [key in T]: unknown } {
  return (
    obj !== null &&
    typeof obj === 'object' &&
    Object.prototype.hasOwnProperty.call(obj, prop)
  );
}

/**
 * Safely checks if an object has a string property.
 */
export function isStringProperty<T extends string>(
  obj: unknown,
  prop: T,
): obj is { [key in T]: string } {
  return hasProperty(obj, prop) && typeof obj[prop] === 'string';
}

/**
 * Safely checks if an object has an object property.
 */
export function isObjectProperty<T extends string>(
  obj: unknown,
  prop: T,
): obj is { [key in T]: object } {
  return (
    hasProperty(obj, prop) &&
    obj[prop] !== null &&
    typeof obj[prop] === 'object'
  );
}
