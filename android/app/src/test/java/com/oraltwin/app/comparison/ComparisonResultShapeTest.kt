package com.oraltwin.app.comparison

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Modifier

/**
 * Asserts the SHAPE of ComparisonResult.Refused, not its behaviour.
 *
 * Refused must expose no numeric property. Phase 5 of the Python project
 * printed raw pixel areas beside a "partly out of frame" label, and a reader
 * could divide them and infer a result the system had declined to give. The
 * fix there was editorial. Here it is structural: if there is no number on the
 * type, no screen can print one.
 *
 * A comment saying so would be a promise. This is a check: adding
 * `val areaPx: Int` to Refused fails this test rather than reaching a screen,
 * which is the whole point of writing it by reflection instead of by eye.
 *
 * Java reflection rather than kotlin-reflect, deliberately: it needs no extra
 * dependency, and a test that drags in a library to test a type is a test that
 * can itself fail to build.
 */
class ComparisonResultShapeTest {

    private val numericTypes: Set<Class<*>> = setOf(
        java.lang.Integer.TYPE, java.lang.Long.TYPE, java.lang.Double.TYPE,
        java.lang.Float.TYPE, java.lang.Short.TYPE, java.lang.Byte.TYPE,
        Integer::class.java, java.lang.Long::class.java,
        java.lang.Double::class.java, java.lang.Float::class.java,
        java.lang.Short::class.java, java.lang.Byte::class.java,
        java.lang.Number::class.java, java.math.BigDecimal::class.java,
        java.math.BigInteger::class.java,
    )

    private fun isNumeric(type: Class<*>): Boolean =
        type in numericTypes || java.lang.Number::class.java.isAssignableFrom(type)

    @Test
    fun refused_declares_no_numeric_field() {
        val offenders = ComparisonResult.Refused::class.java.declaredFields
            .filterNot { it.isSynthetic }
            .filterNot { Modifier.isStatic(it.modifiers) }
            .filter { isNumeric(it.type) }
            .map { "${it.name}: ${it.type.simpleName}" }

        assertTrue(
            "ComparisonResult.Refused must carry no numeric field, so that no " +
                "screen can print a measurement for a comparison that was " +
                "refused. Found: $offenders",
            offenders.isEmpty(),
        )
    }

    @Test
    fun refused_exposes_no_numeric_getter() {
        // a computed property has no backing field, so the fields check above
        // would miss it; getters catch that
        val offenders = ComparisonResult.Refused::class.java.declaredMethods
            .filterNot { it.isSynthetic }
            .filterNot { Modifier.isStatic(it.modifiers) }
            .filter { Modifier.isPublic(it.modifiers) }
            .filter { it.parameterCount == 0 }
            .filterNot { it.name == "hashCode" }   // Int by contract, not a measurement
            .filter { isNumeric(it.returnType) }
            .map { "${it.name}(): ${it.returnType.simpleName}" }

        assertTrue(
            "ComparisonResult.Refused must expose no numeric accessor. " +
                "Found: $offenders",
            offenders.isEmpty(),
        )
    }

    @Test
    fun refused_carries_exactly_the_code_and_the_guidance() {
        val names = ComparisonResult.Refused::class.java.declaredFields
            .filterNot { it.isSynthetic }
            .filterNot { Modifier.isStatic(it.modifiers) }
            .map { it.name }
            .sorted()
        assertEquals(listOf("code", "guidance"), names)
    }

    @Test
    fun the_shape_check_is_not_vacuous() {
        // Comparable DOES carry a number, and the same predicate finds it.
        // Without this, a bug in isNumeric() would make every assertion above
        // pass for the wrong reason.
        val found = ComparisonResult.Comparable::class.java.declaredFields
            .filterNot { it.isSynthetic }
            .filterNot { Modifier.isStatic(it.modifiers) }
            .filter { isNumeric(it.type) }
            .map { it.name }

        assertTrue(
            "the detector should find confidence on Comparable; if it does " +
                "not, the Refused assertions above prove nothing",
            found.contains("confidence"),
        )
    }

    @Test
    fun comparison_result_has_exactly_two_variants() {
        // a third variant would need its own numeric policy, and this test is
        // where that decision gets noticed
        val subclasses = ComparisonResult::class.sealedSubclasses.map { it.simpleName }
        assertEquals(setOf("Comparable", "Refused"), subclasses.toSet())
    }
}
