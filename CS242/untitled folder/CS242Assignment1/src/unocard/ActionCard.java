package unocard;
import game.SpecialEffectsManager;

public abstract class ActionCard extends UnoCard{
	
	private final Color color;
    private final Content content;
    
    public enum Color {
        BLUE,
        RED,
        GREEN,
        YELLOW
    }
	
	public enum Content {
		SKIP, 
		REVERSE,
		DRAWTWO
	}
	
	public ActionCard(Color color, Content content) {
		this.color = color;
		this.content = content;
	}
	
	@Override
    public String getContent() {
    	return this.content.toString();
    }
    
    @Override
    public String getColor() {
    	return this.color.toString();
    }
    
    @Override
    public boolean doMatch(UnoCard ucard) {
    	if(ucard.getColor().compareTo("BLACK") == 0) {
    		return true;
    	}
    	else if(ucard.getColor().compareTo(this.getColor()) == 0) {
    		return true;
    	}
    	else if(ucard.getContent().compareTo(this.getContent()) == 0) {
    		return true;
    	}
    	else {
    		return false;
    	}
    }
    
    /**
     * apply the special effect of the action card.
     * @param effManager: the object that controls the game effects.
     */
    public abstract void doSpecialEffect(SpecialEffectsManager effManager);
    
}
