@file:Suppress("UnstableApiUsage")

import org.gradle.api.tasks.Delete
import org.gradle.jvm.tasks.Jar

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    application
    base
}

// Apply our model setup plugin
apply<ModelSetupPlugin>()

// Project configuration
group = "ai.noesisreality"
version = "1.0.0"

repositories {
    mavenCentral()
    
    // JitPack for FlatBuffers fork from GitHub
    maven { 
        url = uri("https://jitpack.io") 
        content {
            includeGroup("com.github.ariawisp")
        }
    }
}

// Build configuration
val useLocalFlatBuffers: Boolean = project.hasProperty("useLocalFlatBuffers") || 
    file("../flatbuffers").exists()

// Build configuration for FlatBuffers
val flatbuffersForkVersion = libs.versions.flatbuffers.fork.get()

dependencies {
    implementation(libs.clikt)
    implementation(libs.coroutines.core)
    
    // Keep JSON for configuration and non-performance-critical data
    implementation(libs.serialization.json)
    
    // FlatBuffers - Fork with enhanced performance features
    if (useLocalFlatBuffers) {
        // Local development: use built JAR from sibling directory (Kotlin includes Java classes)
        implementation(files("../flatbuffers/kotlin/flatbuffers-kotlin/build/libs/flatbuffers-kotlin-jvm-2.0.0-SNAPSHOT.jar"))
    } else {
        // Production: use JitPack to build from GitHub
        implementation(libs.flatbuffers.kotlin)
    }
    
    testImplementation(libs.kotlin.test)
    testImplementation(libs.coroutines.test)
}

application {
    mainClass.set("ai.noesisreality.cli.NoesisCLIKt")
    applicationName = "noesis"
}

kotlin {
    jvmToolchain(17)
    compilerOptions {
        freeCompilerArgs.addAll(
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
            "-opt-in=kotlinx.serialization.ExperimentalSerializationApi"
        )
    }
}

tasks.test {
    useJUnitPlatform()
}

// Note: FlatBuffers integration configured after task definitions below

// Build configuration
object BuildPaths {
    val swiftBuildDir = "swift-implementation/.build/release"
    val rustBuildDir = "rust-harmony-jni/target/release"
    val jniLibDir = "src/main/resources/native"
    
    // Native libraries
    val swiftLib = "$swiftBuildDir/libNoesisBridge.dylib"
    val rustLib = "$rustBuildDir/libopenai_harmony.dylib"
    val jniSwiftLib = "$jniLibDir/libnoesis_inference.dylib"
    val jniRustLib = "$jniLibDir/libopenai_harmony.dylib"
}

// FlatBuffers code generation using Fork
tasks.register<Exec>("generateFlatBuffers") {
    description = "Generate Kotlin code from FlatBuffers schemas using Fork"
    group = "codegen"
    
    val schemaDir = file("schemas")
    val outputDir = file("src/main/kotlin/ai/noesisreality/protocol/generated")
    
    // Proper incremental build support
    inputs.dir(schemaDir)
    inputs.file("build.gradle.kts") // Rebuild if build config changes
    outputs.dir(outputDir)
    outputs.cacheIf { true } // Enable build cache
    
    doFirst {
        outputDir.mkdirs()
        logger.lifecycle("🔧 Generating FlatBuffers code using Fork...")
    }
    
    // Try flatc from fork first, then system flatc
    val flatcPaths = listOf(
        "../flatbuffers/flatc",                           // Local fork
        "./flatbuffers-fork/flatc",                       // Local fork build
        "${System.getenv("HOME")}/flatbuffers-fork/flatc", // User home install  
        "flatc"                                            // System PATH
    )
    
    val flatcPath = flatcPaths.firstOrNull { File(it).canExecute() || it == "flatc" }
    
    if (flatcPath != null) {
        // Generate Kotlin code with fork enhancements
        commandLine(
            flatcPath, 
            "--kotlin-kmp",         // Use Kotlin multiplatform for correct imports
            "--gen-mutable",         // Generate mutable accessors
            "-o", outputDir.absolutePath,
            *schemaDir.listFiles { _, name -> name.endsWith(".fbs") }
                ?.map { it.absolutePath }?.toTypedArray() ?: emptyArray()
        )
    }
    
    // Enhanced error handling for fork
    doLast {
        val result = executionResult.get()
        if (result.exitValue != 0) {
            logger.error("FlatBuffers code generation failed!")
            logger.error("Solutions:")
            logger.error("1. Clone FlatBuffers fork to ../flatbuffers")
            logger.error("2. Build flatc: cd flatbuffers && cmake -G \"Unix Makefiles\" -DCMAKE_BUILD_TYPE=Release && make -j8")
            logger.error("3. Install system flatc: brew install flatbuffers (or apt-get install flatbuffers-compiler)")
            logger.error("4. Set up PATH to include fork flatc binary")
        } else {
            logger.lifecycle("✅ FlatBuffers code generated successfully with Fork")
        }
    }
    
    isIgnoreExitValue = true  // Don't fail build if flatc missing
}

// Configure FlatBuffers integration after task definitions
if (useLocalFlatBuffers) {
    tasks.named("compileKotlin") {
        dependsOn("buildFlatBuffersFork", "generateFlatBuffers")
    }
    tasks.named("generateFlatBuffers") {
        dependsOn("buildFlatBuffersFork")
    }
} else {
    tasks.named("compileKotlin") {
        dependsOn("generateFlatBuffers")
    }
}

// Build local FlatBuffers fork (only when using local)
tasks.register<Exec>("buildFlatBuffersFork") {
    description = "Build local FlatBuffers fork when useLocalFlatBuffers=true"
    group = "setup"
    
    // Only register this task if we're using local flatbuffers
    onlyIf { useLocalFlatBuffers }
    
    val flatbuffersDir = file("../flatbuffers")
    val kotlinJar = file("../flatbuffers/kotlin/flatbuffers-kotlin/build/libs/flatbuffers-kotlin-jvm-2.0.0-SNAPSHOT.jar")
    
    inputs.dir("../flatbuffers/kotlin/flatbuffers-kotlin/src")
    inputs.file("../flatbuffers/kotlin/build.gradle.kts")
    outputs.file(kotlinJar)
    
    doFirst {
        if (!flatbuffersDir.exists()) {
            throw GradleException(
                """Local flatbuffers directory not found at: ${flatbuffersDir.absolutePath}
                   |Either:
                   |1. Clone FlatBuffers fork to ../flatbuffers, or
                   |2. Remove -PuseLocalFlatBuffers to use GitHub version via JitPack
                """.trimMargin()
            )
        }
        logger.lifecycle("🔧 Building local FlatBuffers fork...")
    }
    
    // Build Kotlin library (includes all FlatBuffers functionality)
    commandLine("bash", "-c", """
        set -e  # Exit on any error
        
        echo "📦 Building FlatBuffers fork Kotlin library locally..."
        
        # Check if JVM JAR already exists
        if [ -f "../flatbuffers/kotlin/flatbuffers-kotlin/build/libs/flatbuffers-kotlin-jvm-2.0.0-SNAPSHOT.jar" ]; then
            echo "✅ FlatBuffers fork JVM JAR already exists, skipping build"
            ls -la ../flatbuffers/kotlin/flatbuffers-kotlin/build/libs/flatbuffers-kotlin-jvm-2.0.0-SNAPSHOT.jar
            exit 0
        fi
        
        # Ensure flatc compiler is built
        if [ ! -f "../flatbuffers/flatc" ]; then
            echo "🔧 Building flatc compiler..."
            cd ../flatbuffers
            cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
                  -DFLATBUFFERS_BUILD_TESTS=OFF \
                  -DFLATBUFFERS_BUILD_FLATHASH=OFF
            make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) flatc
            cd -
        fi
        
        # Build only JVM target to avoid yarn issues
        echo "🦄 Building Kotlin JVM library..."
        cd ../flatbuffers/kotlin
        ./gradlew flatbuffers-kotlin:jvmJar -x test
        cd -
        
        echo "✅ FlatBuffers fork Kotlin library built successfully"
        ls -la ../flatbuffers/kotlin/flatbuffers-kotlin/build/libs/*.jar || true
    """.trimIndent())
}

// Native library build tasks
tasks {
    val buildSwiftBridge by registering(Exec::class) {
        description = "Builds Swift Metal inference JNI bridge"
        group = "native"
        
        workingDir = file("swift-implementation")
        commandLine("swift", "build", "--product", "NoesisBridge", "-c", "release")
        
        inputs.dir("swift-implementation/Sources")
        outputs.file(BuildPaths.swiftLib)
    }
    
    val buildRustHarmonyBridge by registering(Exec::class) {
        description = "Builds Rust Harmony encoding JNI bridge (REQUIRED)"
        group = "native"
        
        workingDir = file("rust-harmony-jni")
        commandLine("cargo", "build", "--release")
        
        inputs.dir("rust-harmony-jni/src")
        inputs.file("rust-harmony-jni/Cargo.toml")
        outputs.file(BuildPaths.rustLib)
    }
    
    val createNativeLibrary by registering {
        description = "Copies native libraries to JNI resource directory"
        group = "native"
        
        dependsOn(buildSwiftBridge, buildRustHarmonyBridge)
        
        inputs.file(BuildPaths.swiftLib)
        inputs.file(BuildPaths.rustLib)
        outputs.files(BuildPaths.jniSwiftLib, BuildPaths.jniRustLib)
        
        doLast {
            val jniLibDir = file(BuildPaths.jniLibDir).apply { mkdirs() }
            
            // Swift inference library
            val swiftLib = file(BuildPaths.swiftLib)
            val jniSwiftLib = file(BuildPaths.jniSwiftLib)
            
            require(swiftLib.exists()) {
                "Swift inference library not found: ${swiftLib.absolutePath}"
            }
            
            swiftLib.copyTo(jniSwiftLib, overwrite = true)
            logger.lifecycle("✅ Swift inference library: ${jniSwiftLib.absolutePath}")
            
            // Rust Harmony library (REQUIRED)
            val rustLib = file(BuildPaths.rustLib)
            val jniRustLib = file(BuildPaths.jniRustLib)
            
            require(rustLib.exists()) {
                """
                ❌ CRITICAL: Rust Harmony library not found at ${rustLib.absolutePath}
                Harmony encoding is required for GPT-OSS models.
                Build with: ./gradlew buildRustHarmonyBridge
                """.trimIndent()
            }
            
            rustLib.copyTo(jniRustLib, overwrite = true)
            logger.lifecycle("✅ Rust Harmony library: ${jniRustLib.absolutePath}")
        }
    }
    
    val buildNativeLibrary by registering {
        description = "Builds all native libraries for JNI bridge"
        group = "native"
        dependsOn(createNativeLibrary)
    }
}

// Main build task
tasks.register("buildNoesis") {
    description = "Complete Noesis build: flatbuffers, models, native libraries, and Kotlin CLI"
    group = "build"
    
    if (useLocalFlatBuffers) {
        dependsOn("buildFlatBuffersFork", "setupModel", "buildNativeLibrary", "build")
        // Task ordering for local flatbuffers
        tasks.named("setupModel") { mustRunAfter("buildFlatBuffersFork") }
        tasks.named("buildNativeLibrary") { mustRunAfter("setupModel") }
        tasks.named("build") { mustRunAfter("buildNativeLibrary") }
    } else {
        dependsOn("setupModel", "buildNativeLibrary", "build")
        // Task ordering for GitHub flatbuffers
        tasks.named("buildNativeLibrary") { mustRunAfter("setupModel") }
        tasks.named("build") { mustRunAfter("buildNativeLibrary") }
    }
}

// Cleanup tasks  
tasks.register<Delete>("cleanNoesis") {
    description = "Clean all build artifacts including native libraries"
    group = "build"
    
    delete(
        ".gradle",
        "build",
        "swift-implementation/.build",
        "rust-harmony-jni/target",
        "src/main/resources/native"
    )
}

tasks.register<Delete>("cleanModels") {
    description = "Clean downloaded models (WARNING: removes ~/.noesis/models)"
    group = "build"
    
    doFirst {
        val modelsDir = file("${System.getProperty("user.home")}/.noesis")
        if (modelsDir.exists()) {
            logger.warn("Removing models directory: ${modelsDir.absolutePath}")
        }
    }
    
    delete("${System.getProperty("user.home")}/.noesis")
}

// Enhanced JAR with proper metadata
tasks.named<Jar>("jar") {
    archiveBaseName.set("noesis")
    archiveVersion.set(project.version.toString())
    
    manifest {
        attributes(
            "Implementation-Title" to "Noesis GPT-OSS CLI",
            "Implementation-Version" to project.version,
            "Implementation-Vendor" to "Noesis Reality LLC",
            "Main-Class" to "ai.noesisreality.cli.NoesisCLIKt"
        )
    }
}

// Configure application scripts with proper JNI library path
tasks.named<CreateStartScripts>("startScripts") {
    doLast {
        val unixScript = file("$outputDir/noesis")
        if (unixScript.exists()) {
            val content = unixScript.readText().replace(
                "exec \"\$JAVACMD\" \"\$@\"",
                "exec \"\$JAVACMD\" -Djava.library.path=\"\$APP_HOME/lib\" \"\$@\""
            )
            unixScript.writeText(content)
            logger.info("Enhanced start script with JNI library path")
        }
    }
}

// Help task
tasks.register("helpNoesis") {
    description = "Show available Noesis-specific tasks"
    group = "help"
    
    doLast {
        println("""
        🚀 Noesis GPT-OSS Build Tasks:
        
        Build Tasks:
          buildNoesis          - Complete build (flatbuffers + models + native + CLI)
          build                - Standard Gradle build
          buildNativeLibrary   - Build Swift + Rust native libraries
          buildSwiftBridge     - Build Swift Metal inference bridge only  
          buildRustHarmonyBridge - Build Rust Harmony encoding bridge only
          buildFlatBuffersFork     - Build local flatbuffers fork (when using local)
          
        Clean Tasks:
          clean                - Clean Gradle build directory
          cleanNoesis          - Clean all build artifacts
          cleanModels          - Remove downloaded models (WARNING)
          
        Model Tasks:
          setupModel           - Download and setup GPT-OSS models
          
        Run Tasks:
          run                  - Run Noesis CLI
          installDist          - Create distribution with start scripts
        
        FlatBuffers Configuration:
          Current mode: ${if (useLocalFlatBuffers) "LOCAL" else "GITHUB"}
          ${if (useLocalFlatBuffers) "Using ../flatbuffers directory" else "Using JitPack from GitHub"}
          
        Usage Examples:
          ./gradlew buildNoesis                    # Full build (GitHub flatbuffers)
          ./gradlew buildNoesis -PuseLocalFlatBuffers # Build with local flatbuffers
          ./gradlew run --args="generate 'Hello'"  # Run CLI
          ./gradlew installDist                    # Create distribution
        """.trimIndent())
    }
}