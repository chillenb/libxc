
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m11_l_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.997069599615066e+01, -2.997094849072053e+01, -2.997220695598736e+01, -2.996846756187195e+01, -2.997040781557484e+01, -2.997040781557484e+01, -2.991062294408143e+00, -2.991163651400405e+00, -2.993983825968319e+00, -2.992985065881088e+00, -2.992828434652734e+00, -2.992828434652734e+00, -6.661692111241879e-01, -6.656187074937596e-01, -6.489626534776102e-01, -6.578755623244417e-01, -6.557698576850531e-01, -6.557698576850531e-01, -3.002045132440118e-01, -3.009902102005526e-01, -6.720196882507530e-01, -2.623266559794910e-01, -2.752896995039857e-01, -2.752896995039818e-01, -4.093115789859263e-02, -4.305549413706712e-02, -2.135047930203302e-01, -2.371006337066866e-02, -2.976059621162321e-02, -2.976059621162321e-02, -7.221776012046622e+00, -7.214565416033498e+00, -7.221591238502601e+00, -7.215224662982344e+00, -7.218110305214304e+00, -7.218110305214304e+00, -1.687587177703755e+00, -1.704008685313899e+00, -1.676179273535091e+00, -1.689324961550827e+00, -1.702616115362844e+00, -1.702616115362844e+00, -5.761910948523921e-01, -6.680467680709462e-01, -5.261008669636527e-01, -5.452082207736113e-01, -5.853880738472532e-01, -5.853880738472532e-01, -2.740889424929600e-01, -3.074549612727161e-01, -2.769588058791803e-01, -2.030830839168556e+00, -2.359979863964758e-01, -2.359979863964772e-01, -1.830259721490255e-02, -2.317854637905642e-02, -1.770751900206161e-02, -2.697799246424282e-01, -2.133520507140531e-02, -2.133520507140511e-02, -3.866079033937349e-01, -5.902999572671324e-01, -5.384964772302457e-01, -4.776451571939504e-01, -5.100413527278118e-01, -5.100413527278118e-01, -3.754891123283635e-01, -5.064975700812097e-01, -4.876171717328784e-01, -5.208706696369743e-01, -4.891912992983961e-01, -4.891912992983973e-01, -6.903396216491047e-01, -3.499480166287713e-01, -3.743902162285385e-01, -3.885495152044023e-01, -3.830488632752583e-01, -3.830488632752592e-01, -4.755175993581099e-01, -2.072229226043996e-01, -2.578346695940387e-01, -3.583065202498642e-01, -2.468791042110384e-01, -2.468791042110386e-01, -5.761574988757219e-02, -6.202679076271540e-03, -1.301290346202890e-02, -2.508175964945332e-01, -1.974214122003319e-02, -1.974214122003325e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m11_l_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.376042841049917e+01, -4.375921718917167e+01, -4.375487908480632e+01, -4.377281053413392e+01, -4.376326973284156e+01, -4.376326973284156e+01, -4.594691213028409e+00, -4.596431246023337e+00, -4.643334660639535e+00, -4.631053668597058e+00, -4.629221230523412e+00, -4.629221230523412e+00, -8.448403321335652e-01, -8.484114411402214e-01, -8.937265234694300e-01, -8.653236239323615e-01, -8.740201676338695e-01, -8.740201676338695e-01, -2.804692938020639e-01, -2.669073165611957e-01, -1.287652133831263e+00, -2.691345226463064e-01, -3.101406936817401e-01, -3.101406936817424e-01, -5.410829293195835e-02, -5.684845406612111e-02, -2.178593482864444e-01, -3.152985050275111e-02, -3.949113108899111e-02, -3.949113108899111e-02, -6.224494912109795e+00, -6.111023653447625e+00, -6.218496370304708e+00, -6.118378784064888e+00, -6.167524249699098e+00, -6.167524249699098e+00, -2.386657889244936e+00, -2.488750081340587e+00, -2.331864186762919e+00, -2.413060166622968e+00, -2.473857636047242e+00, -2.473857636047242e+00, -7.495720801128637e-01, -1.077686020467940e+00, -6.533453154723827e-01, -8.372041777841643e-01, -7.921063843818837e-01, -7.921063843818837e-01, 5.905404713503883e-02, -3.466517119625995e-01, 6.632091409134845e-02, -3.830173652121028e+00, -1.382949857200413e-01, -1.382949857200343e-01, -2.436170831374122e-02, -3.082211814687963e-02, -2.351271583926636e-02, -8.635851124711855e-02, -2.833393808511851e-02, -2.833393808511769e-02, -3.778400443265659e-02, -3.580737893381323e-01, -1.584241666912012e-01, -4.648306125033207e-02, -9.372442730087187e-02, -9.372442730087187e-02, 1.968621689770712e-02, -4.492213983745672e-01, -5.416115420849981e-01, -9.835182281330916e-01, -7.325285771717882e-01, -7.325285771717922e-01, -1.103584941401428e+00, -3.410232099613315e-01, -3.012513495547739e-01, -3.200018506622980e-01, -2.852843489419758e-01, -2.852843489419818e-01, -4.461208285567068e-01, -2.196352448075196e-01, -2.050562014010487e-01, -2.857833200053429e-01, 5.244109650110303e-02, 5.244109650111035e-02, -7.571150838822689e-02, -8.266988058712468e-03, -1.732678206423072e-02, 3.895086061978574e-02, -2.622077408079945e-02, -2.622077408079852e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.363708533564433e-08, -3.363904866533642e-08, -3.364644424988331e-08, -3.361740813258274e-08, -3.363280598806422e-08, -3.363280598806422e-08, 7.934262295218960e-07, 7.862933404873982e-07, 5.969648343322202e-07, 6.620016062812825e-07, 6.598457435460987e-07, 6.598457435460987e-07, -1.527992422799304e-03, -1.542189908244833e-03, -2.287141883444575e-03, -2.082051675575475e-03, -2.101055279276512e-03, -2.101055279276512e-03, -9.000728593406470e-01, -1.019684575648690e+00, -1.842121361117452e-03, 1.783587118620128e-01, -1.412107480851606e-01, -1.412107480851448e-01, -5.607400337118660e+01, -5.924253045726068e+01, -3.353872348722173e+01, -5.094299296905714e+01, -6.472515168458938e+01, -6.472515168458956e+01, -1.883152617357456e-05, -1.912464018680160e-05, -1.884792641361994e-05, -1.910650223298344e-05, -1.897796647068785e-05, -1.897796647068785e-05, 5.145836674490190e-06, 4.311826658639835e-08, 7.521540762345406e-06, 3.139573294908960e-06, 9.160299521975667e-07, 9.160299521975667e-07, -5.498011153108412e-03, 9.865037935001448e-03, -2.161472370997629e-02, -4.801354297872279e-03, -1.119002594900957e-03, -1.119002594900957e-03, -1.081959691474930e+01, -3.835716629854629e-02, -1.494983652462808e+01, -1.632703335850808e-04, -2.162371388093342e+00, -2.162371388093354e+00, -6.712039845571312e+01, -5.995582218871570e+01, -3.764492879073387e+02, -3.660005931443326e+01, -1.752009813824669e+02, -1.752009813824661e+02, -2.436680847950261e-01, -8.227191158486490e-02, -1.254018420709709e-01, -1.713611709508389e-01, -1.470544253976503e-01, -1.470544253976503e-01, -2.317482105889182e-01, -2.330764529760703e-02, 7.283404419440792e-03, 3.551649589615304e-02, 2.527092888212896e-02, 2.527092888212892e-02, 5.481308078962784e-03, -2.551986897855716e-01, -2.826625102088627e-01, -1.847365101736801e-01, -2.559786985604556e-01, -2.559786985604559e-01, -4.722969983520723e-02, -3.057148280877751e+01, -3.064312100509444e+01, -2.184641758716539e-01, -2.690170440032071e+01, -2.690170440032069e+01, -4.522178043334142e+01, -2.941844495374960e+02, -1.423770343082668e+02, -3.176251758181642e+01, -2.212858877824152e+02, -2.212858877824158e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m11_l_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m11_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.962260494876038e-03, 9.961937768470685e-03, 9.960963412590363e-03, 9.965752783049234e-03, 9.963178994075153e-03, 9.963178994075153e-03, 4.358891762438272e-03, 4.375466611390124e-03, 4.821575726100694e-03, 4.693186780744964e-03, 4.685040748969053e-03, 4.685040748969053e-03, 1.524818506947228e-03, 2.038340301236550e-03, 1.164428479609321e-02, 8.121939173636662e-03, 8.963561067618172e-03, 8.963561067618172e-03, 3.451518613058897e-02, 3.771878084956116e-02, 2.059544992008277e-02, -4.796732301392835e-02, -1.241304737567570e-02, -1.241304737567252e-02, -8.746128690344491e-05, -1.062107644762462e-04, -7.950088755847243e-03, -1.239578661756051e-05, -3.420071037060262e-05, -3.420071037060262e-05, 1.686556246473674e-02, 1.606513517917414e-02, 1.682995773696373e-02, 1.612358405177670e-02, 1.646144523408342e-02, 1.646144523408342e-02, 2.775336985629815e-03, 5.293267458503737e-03, 1.588446428140457e-03, 3.589643488500520e-03, 4.869430447877846e-03, 4.869430447877846e-03, 3.130100644305690e-03, 8.384655729188735e-02, 3.211927356421637e-02, 5.308166456442196e-02, -3.207072846222794e-03, -3.207072846222794e-03, -5.707121973007333e-02, -1.206348826372014e-02, -5.303504415827378e-02, 5.829779438988888e-02, -6.063108402017771e-02, -6.063108402019399e-02, -3.942890214898629e-06, -1.023849059686949e-05, -3.695476607081404e-05, -3.016816801805297e-02, -2.480755497151672e-05, -2.480755497129595e-05, -1.091926283056502e+00, -5.610001414121266e-01, -8.977133600434358e-01, -1.120533195208359e+00, -1.023084538115335e+00, -1.023084538115335e+00, -1.504140778662469e+00, -4.011724810480596e-02, -1.227713121065638e-01, 1.042884705152917e-01, -6.874626124136902e-02, -6.874626124136737e-02, 6.432350006821035e-02, 2.171983482630903e-02, 3.362223180539455e-02, 3.111764854126776e-02, 3.608337567533512e-02, 3.608337567533998e-02, 1.184209152270701e-02, -7.136792635995808e-03, -1.491308374590316e-02, -1.557757417195578e-02, -5.553358779217096e-02, -5.553358779217255e-02, -1.174842193118578e-04, -2.396193931773997e-07, -6.349844759766914e-06, -5.362436806244786e-02, -2.800683866373863e-05, -2.800683866552696e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05