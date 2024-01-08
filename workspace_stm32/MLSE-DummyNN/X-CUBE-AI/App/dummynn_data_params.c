/**
  ******************************************************************************
  * @file    dummynn_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Jan  8 14:34:55 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "dummynn_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_dummynn_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_dummynn_weights_array_u64[71] = {
  0x3cda0a6dbe370721U, 0x3efca0aabf03e35bU, 0x3e8f76213e94f871U, 0x3e73ee07beaed197U,
  0x3ed15f363eea670bU, 0x3d8c8b26be9db76cU, 0xbe8e4409bea90a35U, 0x3eea624d3e8e0c17U,
  0xbe0d87283e9c4c1cU, 0xbe208a233eec4dc3U, 0x3bcf0c7c3e4ddfc1U, 0x3e2fa95d3ca354c9U,
  0xbeca8f5a3ea256e2U, 0xbe52c62b3cc33a38U, 0x3ea82834bed5db6dU, 0xbf0bd7eebe0c3db7U,
  0xbe84f17ebf051229U, 0xbf0efff6bebbd167U, 0xbba19567be3b6491U, 0x3e0b6d51bec25703U,
  0xbd11c241be406c9aU, 0xbe4c638bbec60edeU, 0x3ef7e6f83eb1b83bU, 0x3e141a72bd98dd78U,
  0xbe8bff7cbe1bb6b8U, 0xbe26b12cbdbe1df9U, 0xbe71b38bbceb122bU, 0x3e981d7ebea276c1U,
  0x3e6ad5073ea4036fU, 0xbe975559beb62e74U, 0x3e952701bdff78c2U, 0x3e9ed845bc6cb53dU,
  0xbd12c089be75d93aU, 0x3e41e728bec77023U, 0x3e9dc62ebecdae59U, 0xbe97ab293c45dcbcU,
  0x3e2bf74c3e50b160U, 0x3e926edcbef9e95eU, 0xbe770642be4a9bf2U, 0xbd043d56be1cad37U,
  0x3ed60f753dfc56c7U, 0xbd4857a3bedf3c62U, 0xbe94cb1bbe36cdfaU, 0xbe022c31be8b4b49U,
  0x3e51843abeed021dU, 0xbe17345abe6d31f6U, 0x3ee822c73e73c8d2U, 0x3ec8cbae3ebec026U,
  0xbc55ae33be0a6a70U, 0xbeb3a5ee3cf6aa84U, 0xbde625443cdbf147U, 0x3cbe5be43d4a4dafU,
  0x3ce733a6bdfcd05eU, 0x3d46c87b3dbfb700U, 0xbdc0022ebd98084bU, 0x3bb6f6563db557d4U,
  0xbddd8b12bd55f4caU, 0xbd89a0633dcac87dU, 0xbc63009c3da7e2e2U, 0xbd9d790fbe055659U,
  0xbee508de3de61319U, 0x3e53c207bee70908U, 0xbe93d0ccbee81ffdU, 0xbe057367bef3ee0aU,
  0xb9d6c0fa3eecd2b2U, 0x3f045f99bd879acdU, 0x3e1f80c2bd598898U, 0x3ee7d612bdd51c84U,
  0x3ea8f0dcbe3303ecU, 0x3ee78819be8cefb4U, 0xbd8cf239U,
};


ai_handle g_dummynn_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_dummynn_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

