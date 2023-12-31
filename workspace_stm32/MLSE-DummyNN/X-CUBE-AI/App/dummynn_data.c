/**
  ******************************************************************************
  * @file    dummynn_data.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Jan  8 14:34:55 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#include "dummynn_data.h"
#include "ai_platform_interface.h"

AI_API_DECLARE_BEGIN
ai_buffer g_dummynn_data_map_activations[AI_DUMMYNN_DATA_ACTIVATIONS_COUNT] = {
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 100, 1, 1),
    100, NULL, NULL),    /* heap_overlay_pool */
  };
ai_buffer g_dummynn_data_map_weights[AI_DUMMYNN_DATA_WEIGHTS_COUNT] = {
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 564, 1, 1),
    564, NULL, s_dummynn_weights_array_u64),   /* weights_array */
  };


/*!
 * @brief Get network activations buffer initialized struct.
 * @ingroup dummynn_data
 * @param[in] ptr a pointer to the activations array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_dummynn_data_activations_buffer_get(const ai_handle ptr)
{
  ai_buffer buf = AI_BUFFER_INIT(
    AI_FLAG_NONE, AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_DUMMYNN_DATA_ACTIVATIONS_SIZE, 1, AI_DUMMYNN_DATA_ACTIVATIONS_COUNT),
    AI_DUMMYNN_DATA_ACTIVATIONS_SIZE,
    NULL, ptr);
  return buf;
}

/*!
 * @brief Get network weights buffer initialized struct.
 * @ingroup dummynn_data
 * @param[in] ptr a pointer to the weights array storage area
 * @return an ai_buffer initialized struct
 */
AI_DEPRECATED
AI_API_ENTRY
ai_buffer ai_dummynn_data_weights_buffer_get(const ai_handle ptr)
{
  ai_buffer buf = AI_BUFFER_INIT(
    AI_FLAG_NONE, AI_BUFFER_FORMAT_U8|AI_BUFFER_FMT_FLAG_CONST,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_DUMMYNN_DATA_WEIGHTS_SIZE, 1, AI_DUMMYNN_DATA_WEIGHTS_COUNT),
    AI_DUMMYNN_DATA_WEIGHTS_SIZE,
    NULL, ptr);
  return buf;
}


/*!
 * @brief Get network weights array pointer as a handle ptr.
 * @ingroup dummynn_data
 * @return a ai_handle pointer to the weights array
 */
AI_DEPRECATED
AI_API_ENTRY
ai_handle ai_dummynn_data_weights_get(void)
{
  return AI_HANDLE_PTR(g_dummynn_weights_table);

}


/*!
 * @brief Get network params configuration data structure.
 * @ingroup dummynn_data
 * @return true if a valid configuration is present, false otherwise
 */
AI_API_ENTRY
ai_bool ai_dummynn_data_params_get(ai_network_params* params)
{
  if (!params) return false;
  
  const ai_buffer_array map_activations = 
    AI_BUFFER_ARRAY_OBJ_INIT(AI_FLAG_NONE, AI_DUMMYNN_DATA_ACTIVATIONS_COUNT, g_dummynn_data_map_activations);
  
  const ai_buffer_array map_weights = 
    AI_BUFFER_ARRAY_OBJ_INIT(AI_FLAG_NONE, AI_DUMMYNN_DATA_WEIGHTS_COUNT, g_dummynn_data_map_weights);

  return ai_platform_bind_network_params(params, &map_weights, &map_activations);
}


AI_API_DECLARE_END
