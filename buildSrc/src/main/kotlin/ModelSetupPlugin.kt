import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.*

/**
 * Plugin for setting up Noesis models with automatic directory creation and download
 */
class ModelSetupPlugin : Plugin<Project> {
    
    override fun apply(project: Project) {
        project.tasks.register<ModelSetupTask>("setupModel") {
            description = "Sets up model directory and downloads GPT-OSS model"
            group = "noesis"
        }
    }
}